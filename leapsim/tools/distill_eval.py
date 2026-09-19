"""distill_eval.py — Phase D: evaluate the distilled student on the angvel harness.

Drives the LeapHandRot env with the BC student and lets the env compute the
SAME per-category angular-velocity metric used for B and C·z1 (the env prints
`mean object angvel ... | per-category: {...}` from reset_idx whenever
`+task.env.print_object_angvel=true`). Using the env's own metric — rather than
the rl_games player — keeps the number identical to the crosscat runs while
letting us load an MLP student we built ourselves. (The student is also loadable
by the stock player via `test=true train=LeapHandRotBC checkpoint=...`; this
custom driver is the primary path because it is self-contained and robust.)

Eval regime matches crosscat: fixed scale 1.0, dynamics OFF, stochastic-capable.
The student is DETERMINISTIC (mu) by default — BC regresses to the specialist
mean, so mu IS the student's behaviour; pass `+eval.stochastic=true` to instead
sample with the teacher-calibrated sigma (parity with B/C's stochastic eval).

Run from src/LEAP_Hand_Sim/leapsim. Example (zero-shot cuboid, CPU smoke):

  uv run python tools/distill_eval.py task=LeapHandRot headless=true \
    wandb_activate=false log_to_sheet=false pipeline=cpu sim_device=cpu \
    rl_device=cpu graphics_device_id=-1 num_envs=64 \
    task.env.object.type=cuboid_train \
    task.env.grasp_cache_name=leap_hand_in_cuboid_train task.env.z_mode=z1 \
    task.env.randomization.randomizeMass=False \
    task.env.randomization.randomizeCOM=False \
    task.env.randomization.randomizeFriction=False \
    task.env.randomization.randomizeScaleList=[1.0] +task.env.scale_list_jitter=0 \
    +task.env.print_object_angvel=true \
    checkpoint=runs/distill_student_z1_smoke/nn/distill_student_z1_smoke.pth \
    +eval.games=200
"""
import isaacgym  # noqa: F401  (MUST precede torch)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make tools/ importable

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

import leapsim
import distill_lib as D


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig) -> None:
    student_ckpt = cfg.checkpoint
    assert student_ckpt, "pass checkpoint=<student .pth>"
    student_ckpt = hydra.utils.to_absolute_path(student_ckpt)
    bc_config = OmegaConf.select(cfg, "eval.bc_config") or os.path.join(
        os.path.dirname(__file__), "..", "cfg", "train", "LeapHandRotBC.yaml")
    games_target = int(OmegaConf.select(cfg, "eval.games") or 768)
    max_steps = int(OmegaConf.select(cfg, "eval.max_steps") or 6000)
    stochastic = bool(OmegaConf.select(cfg, "eval.stochastic") or False)

    device = cfg.rl_device
    env = leapsim.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device,
                       cfg.rl_device, cfg.graphics_device_id, cfg.headless,
                       cfg.multi_gpu, cfg.capture_video, cfg.force_render, cfg)
    assert "print_object_angvel" in env.env_cfg, \
        "pass +task.env.print_object_angvel=true so the env computes per-category angvel"
    obs_dim = env.num_obs

    # Build the MLP student from the BC config (decoupled from the train group so
    # a stray train=LeapHandRotPPO can't silently build a recurrent net).
    with open(os.path.abspath(bc_config)) as f:
        bc_params = yaml.safe_load(f)["params"]
    assert "rnn" not in bc_params["network"], "eval expects the MLP student config"
    model = D.build_a2c_model(bc_params, obs_dim=obs_dim, actions_num=env.num_actions,
                              num_seqs=1, device=device,
                              normalize_input=True, normalize_value=True)
    D.load_model_weights(model, student_ckpt, device)
    model.eval()
    sig = model.a2c_network.sigma.detach().exp().mean().item()
    print(f"[eval] student {os.path.basename(student_ckpt)} obs_dim={obs_dim} "
          f"z_mode={env.z_mode}; driver={'stochastic sigma%.3f' % sig if stochastic else 'deterministic (mu)'}; "
          f"target games={games_target}")

    env.reset_buf[:] = 1
    obs = env.reset()["obs"].to(device)
    games = 0
    for step in range(max_steps):
        with torch.no_grad():
            res = model({"is_train": False, "prev_actions": None, "obs": obs, "rnn_states": None})
        drive = res["actions"] if stochastic else res["mus"]
        obs, _, done, _ = env.step(torch.clamp(drive, -1.0, 1.0))
        obs = obs["obs"].to(device)
        games += int(done.sum().item())
        if games >= games_target:
            break

    # Final summary from the env's own accumulators (same metric as B / C·z1).
    buf = list(env.object_angvel_finite_diff_ep_buf)
    overall = float(sum(buf) / len(buf)) if buf else float("nan")
    percat = {f: round(s / c, 4) for f, (s, c) in sorted(getattr(env, "_angvel_cat_acc", {}).items())}
    print("=" * 70)
    print(f"[eval RESULT] object.type={cfg.task.env.object.type}  z_mode={env.z_mode}  "
          f"driver={'stochastic' if stochastic else 'deterministic'}")
    print(f"[eval RESULT] episodes={games}  overall angvel={overall:.4f} rad/s  per-category={percat}")
    print("=" * 70)


if __name__ == "__main__":
    main()
