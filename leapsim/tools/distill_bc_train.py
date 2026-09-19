"""distill_bc_train.py — Phase C: behaviour-clone a single z-conditioned student.

Pools the (obs, specialist-mu) pairs from every specialist and fits ONE MLP
student pi(a | proprio, z) by supervised MSE on the specialist actions — the
dense-supervision counterpart to PPO. The student uses the SAME concat-z
injection as the C·z1 policy, so this isolates the *learning-signal* axis: does
dense imitation make the concatenated z usable where sparse RL reward did not?

Runs on CPU with no sim (pure torch + rl_games model builder), so it never
touches the GPU. The student is built through rl_games' own builder from
cfg/train/LeapHandRotBC.yaml, so the saved checkpoint reloads through the stock
player: `test=true train=LeapHandRotBC checkpoint=<student>` — or via
tools/distill_eval.py (custom rollout, the primary eval here).

Design choices (documented in tools/run_distill.sh):
  * MLP student (no GRU): keeps BC a plain per-sample regression. The 3-step obs
    history already supplies short-horizon context. GRU-student is the noted
    upgrade (needs sequence batches + BPTT).
  * regress to the specialist MEAN action mu (clean target); the env was driven
    by the specialist's SAMPLED action during collection (realistic states).
  * student log-sigma is set to the mean specialist log-sigma (NOT trained), so
    a stochastic eval is calibrated to the teacher's exploration noise — the
    stage0 memo shows deterministic eval under-reads this policy family.

Example:
  uv run python tools/distill_bc_train.py \
    --data tools/distill_data/smoke_cuboid0.npz \
    --student-zmode z1 --out runs/distill_student_z1_smoke \
    --epochs 30 --batch 4096 --lr 1e-3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make tools/ importable

import numpy as np
import torch
import yaml

import distill_lib as D

NFRAMES = D.NFRAMES


def load_pool(patterns):
    obs_list, mu_list, logstds, metas = [], [], [], []
    files = []
    for p in patterns:
        files += sorted(glob.glob(p))
    assert files, f"no data files matched {patterns}"
    z_dim = base = full = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if z_dim is None:
            z_dim, base, full = int(d["z_dim"]), int(d["base_per_frame"]), int(d["full_dim"])
        assert int(d["z_dim"]) == z_dim and int(d["full_dim"]) == full, \
            f"{f}: inconsistent obs layout across specialists"
        obs_list.append(d["obs"]); mu_list.append(d["mu"])
        logstds.append(d["teacher_logstd"])
        metas.append(dict(file=os.path.basename(f), n=int(d["obs"].shape[0]),
                          object_type=str(d["object_type"]),
                          object_ids=d["object_ids"].tolist(),
                          teacher=str(d["teacher_ckpt"])))
    obs = np.concatenate(obs_list, 0)
    mu = np.concatenate(mu_list, 0)
    mean_logstd = np.mean(np.stack(logstds, 0), 0).astype(np.float32)
    return obs, mu, z_dim, base, full, mean_logstd, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="npz file(s)/glob(s) from distill_collect")
    ap.add_argument("--student-zmode", choices=["none", "z0", "z1"], default="z1",
                    help="z1 = analytic shape prior; z0 = coarse geometry prefix; "
                         "none = proprio-only control (the clean 3-way mirrors RL B/C/C·z1)")
    ap.add_argument("--out", required=True, help="student run dir (checkpoint -> <out>/nn/<name>.pth)")
    ap.add_argument("--bc-config", default=os.path.join(os.path.dirname(__file__), "..", "cfg", "train", "LeapHandRotBC.yaml"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    obs, mu, data_zdim, base, full, mean_logstd, metas = load_pool(args.data)
    print(f"[bc] pooled {obs.shape[0]} pairs from {len(metas)} specialist file(s); "
          f"data z_dim={data_zdim} base_per_frame={base} full_dim={full}")
    for m in metas:
        print(f"[bc]   {m['file']}: n={m['n']} obj={m['object_type']} ids={m['object_ids']} teacher={m['teacher']}")

    # ── Assemble the student's observation ─────────────────────────────────────
    obs_t = torch.from_numpy(obs).float()
    mu_t = torch.from_numpy(mu).float()
    student_zdim = D.Z_DIMS[args.student_zmode]           # none:0  z0:5  z1:8
    assert student_zdim <= data_zdim, (
        f"student z_mode={args.student_zmode} needs {student_zdim} z-dims but the "
        f"collected data only has {data_zdim} (collect with z_mode=z1)")
    student_obs = D.keep_z(obs_t, data_zdim, student_zdim, base)  # prefix-slice of the z1 obs
    obs_dim = student_obs.shape[1]
    actions_num = mu_t.shape[1]
    print(f"[bc] student z_mode={args.student_zmode} obs_dim={obs_dim} actions={actions_num}")

    # ── Build the student via rl_games (same builder the player uses) ──────────
    with open(os.path.abspath(args.bc_config)) as f:
        bc_params = yaml.safe_load(f)["params"]
    assert "rnn" not in bc_params["network"], "LeapHandRotBC must be MLP-only (no rnn block)"
    model = D.build_a2c_model(bc_params, obs_dim=obs_dim, actions_num=actions_num,
                              num_seqs=1, device=args.device,
                              normalize_input=True, normalize_value=True)

    # Freeze input normalisation to the dataset statistics (eval() stops the
    # RunningMeanStd from drifting during BC; MLP has no other train/eval delta).
    with torch.no_grad():
        m = student_obs.mean(0).double()
        v = student_obs.var(0, unbiased=False).clamp_min(1e-8).double()
        model.running_mean_std.running_mean.copy_(m)
        model.running_mean_std.running_var.copy_(v)
        model.running_mean_std.count.fill_(float(student_obs.shape[0]))
        # calibrate student log-sigma to the mean specialist log-sigma
        model.a2c_network.sigma.copy_(torch.from_numpy(mean_logstd).to(args.device))
    model.eval()  # freeze running_mean_std; grads still flow to the MLP

    # Train only the actor pathway (mu head + shared MLP); sigma frozen (set above).
    train_params = [p for n, p in model.named_parameters()
                    if n.startswith("a2c_network.actor_mlp") or n.startswith("a2c_network.mu")]
    opt = torch.optim.Adam(train_params, lr=args.lr)

    n = student_obs.shape[0]
    idx = torch.randperm(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Ytr = student_obs[tr_idx].to(args.device), mu_t[tr_idx].to(args.device)
    Xva, Yva = student_obs[val_idx].to(args.device), mu_t[val_idx].to(args.device)

    def student_mu(x):
        return model({"is_train": False, "prev_actions": None, "obs": x, "rnn_states": None})["mus"]

    for ep in range(args.epochs):
        perm = torch.randperm(Xtr.shape[0])
        tot, cnt = 0.0, 0
        for b in range(0, Xtr.shape[0], args.batch):
            bi = perm[b:b + args.batch]
            pred = student_mu(Xtr[bi])
            loss = torch.nn.functional.mse_loss(pred, Ytr[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * bi.numel(); cnt += bi.numel()
        with torch.no_grad():
            vloss = torch.nn.functional.mse_loss(student_mu(Xva), Yva).item()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep+1:3d}/{args.epochs}  train_mse={tot/cnt:.5f}  val_mse={vloss:.5f}")

    # ── Save an rl_games-format checkpoint ─────────────────────────────────────
    name = os.path.basename(args.out.rstrip("/"))
    nn_dir = os.path.join(args.out, "nn")
    os.makedirs(nn_dir, exist_ok=True)
    ckpt_path = os.path.join(nn_dir, f"{name}.pth")
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "distill": {"student_zmode": args.student_zmode, "obs_dim": obs_dim,
                            "student_zdim": student_zdim, "base_per_frame": base,
                            "nframes": NFRAMES, "sources": [m["file"] for m in metas]}},
               ckpt_path)
    with open(os.path.join(args.out, "distill_meta.json"), "w") as f:
        json.dump({"student_zmode": args.student_zmode, "obs_dim": obs_dim,
                   "final_val_mse": vloss, "mean_teacher_sigma": float(np.exp(mean_logstd).mean()),
                   "n_pairs": int(n), "sources": metas}, f, indent=2)
    print(f"[bc] saved student -> {ckpt_path}  (val_mse={vloss:.5f}, "
          f"student z_mode={args.student_zmode}, obs_dim={obs_dim})")
    print(f"[bc] eval it with: tools/distill_eval.py ... "
          f"task.env.z_mode={args.student_zmode} checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
