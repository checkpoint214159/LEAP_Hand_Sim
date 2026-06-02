"""rerun_algo_observer.py — rl-games AlgoObserver that writes training scalars to Rerun.

Replaces TensorBoard's SummaryWriter. Owns a single continuous RecordingStream
(scalars.rrd) that runs for the full training lifetime. Each logged value is
annotated with three time coordinates — frame, epoch, wall_time — so the viewer
can scrub on any axis. The frame axis aligns with RerunVisualizer's spatial
windows so both stores can be correlated in the viewer.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import rerun as rr
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext


class RerunAlgoObserver(AlgoObserver):
    """AlgoObserver that logs training scalars to a continuous Rerun recording.

    Args:
        output_dir: Directory to write scalars.rrd into. Typically
            experiment_dir / "rerun", same folder as spatial windows.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir: Path = Path(output_dir)
        self._stream: Optional[rr.RecordingStream] = None
        self._start_time: float = 0.0

        self._ep_infos: List[Dict[str, Any]] = []
        self._direct_info: Dict[str, Any] = {}
        self._mean_scores: Optional[Any] = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # AlgoObserver interface
    # ------------------------------------------------------------------

    def after_init(self, algo: Any) -> None:
        self._device = algo.device
        self._mean_scores = torch_ext.AverageMeter(1, algo.games_to_track).to(algo.ppo_device)
        self._start_time = time.monotonic()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "scalars.rrd"
        self._stream = rr.new_recording(application_id="leap_training", recording_id="scalars", make_default=False)
        rr.save(str(out_path), recording=self._stream)

    def process_infos(self, infos: Dict[str, Any], done_indices: Any) -> None:
        if not isinstance(infos, dict):
            return
        if "episode" in infos:
            self._ep_infos.append(infos["episode"])
        self._direct_info = {
            k: v for k, v in infos.items()
            if isinstance(v, (float, int))
            or (isinstance(v, torch.Tensor) and v.ndim == 0)
        }

    def after_clear_stats(self) -> None:
        if self._mean_scores is not None:
            self._mean_scores.clear()

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        if self._stream is None:
            return

        self._set_time(frame, epoch_num, total_time)

        for ep_info in self._ep_infos:
            for key, val in ep_info.items():
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor([val])
                if val.ndim == 0:
                    val = val.unsqueeze(0)
                rr.log(
                    f"training/episode/{key}",
                    rr.Scalar(float(val.to(self._device).mean())),
                    recording=self._stream,
                )
        self._ep_infos.clear()

        for key, val in self._direct_info.items():
            scalar = float(val) if not isinstance(val, torch.Tensor) else float(val)
            rr.log(f"training/env/{key}", rr.Scalar(scalar), recording=self._stream)

        if self._mean_scores is not None and self._mean_scores.current_size > 0:
            rr.log(
                "training/scores/mean",
                rr.Scalar(float(self._mean_scores.get_mean())),
                recording=self._stream,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_time(self, frame: int, epoch_num: int, total_time: float) -> None:
        rr.set_time_sequence("frame",  frame,      recording=self._stream)
        rr.set_time_sequence("epoch",  epoch_num,  recording=self._stream)
        rr.set_time_seconds("wall_time", total_time, recording=self._stream)
