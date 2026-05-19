from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple
import xml.etree.ElementTree as ET

import numpy as np
import rerun as rr


class RerunFrame(NamedTuple):
    rb_states: np.ndarray   # [N, 13] float32, cpu — world-frame pos+quat for each hand link
    obj_pos: np.ndarray     # [3] float32
    obj_rot: np.ndarray     # [4] float32, xyzw convention
    targets: np.ndarray     # [D] float32 — commanded joint angles
    dof_pos: np.ndarray     # [D] float32 — actual joint angles
    reset: bool
    linvel_mag: float
    angvel_mag: float


class RerunVisualizer:
    """Windowed .rrd recorder for a single observed environment.

    The task calls tick() every control step; this class owns the cadence
    (when to open/close windows) and all rerun API calls.
    """

    def __init__(self, rr_cfg: dict, urdf_path: Path, link_names: List[str]) -> None:
        self._enabled = bool(rr_cfg.get('enabled', False))
        if not self._enabled:
            return

        self._window_length = int(rr_cfg['window_length_steps'])
        self._period        = int(rr_cfg['record_every_n_steps'])
        self._output_dir    = Path(rr_cfg['output_dir'])
        self._fidelity      = rr_cfg.get('fidelity', 'mesh')
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._link_names = list(link_names)
        self._meshes = self._load_meshes(urdf_path) if self._fidelity == 'mesh' else {}

        self._global_step  = 0
        self._window_step  = 0
        self._window_count = 0
        self._in_window    = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, frame: RerunFrame) -> None:
        if not self._enabled:
            return

        self._global_step += 1

        if not self._in_window and self._global_step % self._period == 0:
            self._open_window()
            self._in_window   = True
            self._window_step = 0

        if self._in_window:
            if frame.reset:
                rr.log("events", rr.TextLog(f"reset — global step {self._global_step}"))
            self._log_frame(frame)
            self._window_step += 1
            if self._window_step >= self._window_length:
                self._in_window    = False
                self._window_count += 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_window(self) -> None:
        path = (
            self._output_dir
            / f"window_{self._window_count:04d}_step_{self._global_step:08d}.rrd"
        )
        rr.init("leap_hand", recording_id=path.stem, spawn=False)
        rr.save(str(path))
        print(f"[Rerun] window {self._window_count} → {path.name}")

        for name in self._link_names:
            if name in self._meshes:
                print("[rerun_vis] loading mesh for", name)
                verts, faces = self._meshes[name]
                rr.log(
                    f"world/hand/{name}",
                    rr.Mesh3D(vertex_positions=verts, triangle_indices=faces),
                    static=True,
                )
            else:
                rr.log(
                    f"world/hand/{name}",
                    rr.Boxes3D(half_sizes=[[0.008, 0.008, 0.008]]),
                    static=True,
                )
        rr.log("world/object", rr.Boxes3D(half_sizes=[[0.04, 0.04, 0.04]]), static=True)

    def _log_frame(self, frame: RerunFrame) -> None:
        rr.set_time_sequence("step", self._window_step)

        for i, name in enumerate(self._link_names):
            print("[rerun_vis] logging frame for", name)
            print("[rerun_vis]", frame.rb_states[i], frame.rb_states[i].shape)
            rr.log(
                f"world/hand/{name}",
                rr.Transform3D(
                    translation=frame.rb_states[i, 0:3],
                    rotation=rr.Quaternion(xyzw=frame.rb_states[i, 3:7]),
                ),
            )

        rr.log(
            "world/object",
            rr.Transform3D(
                translation=frame.obj_pos,
                rotation=rr.Quaternion(xyzw=frame.obj_rot),
            ),
        )

        for i in range(len(frame.targets)):
            rr.log(f"control/target/joint_{i:02d}", rr.Scalar(float(frame.targets[i])))
            rr.log(f"control/actual/joint_{i:02d}", rr.Scalar(float(frame.dof_pos[i])))

        rr.log("object/linvel_mag", rr.Scalar(frame.linvel_mag))
        rr.log("object/angvel_mag", rr.Scalar(frame.angvel_mag))

    # ------------------------------------------------------------------
    # Asset loading (mesh fidelity only)
    # ------------------------------------------------------------------

    def _load_meshes(self, urdf_path: Path) -> dict:
        """Parse URDF; return dict[link_name → (verts, faces)] with visual origin baked in."""
        asset_dir = urdf_path.parent
        tree = ET.parse(str(urdf_path))
        stl_cache: dict = {}
        meshes: dict = {}
        for link in tree.getroot().findall('link'):
            name = link.get('name')
            visual = link.find('visual')
            if visual is None:
                continue
            origin = visual.find('origin')
            if origin is not None:
                xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
            else:
                xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            mesh_elem = visual.find('geometry/mesh')
            if mesh_elem is None:
                continue
            filename = mesh_elem.get('filename')
            if filename not in stl_cache:
                stl_cache[filename] = self._load_stl(asset_dir / filename)
            raw_verts, faces = stl_cache[filename]
            meshes[name] = (self._apply_visual_origin(raw_verts, xyz, rpy), faces)
        return meshes

    @staticmethod
    def _load_stl(path: Path):
        """Return (vertex_positions, triangle_indices) from a binary STL file."""
        with open(path, 'rb') as f:
            f.seek(80)
            count = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
            raw = f.read(count * 50)
        # 50-byte record: 12 bytes normal | 36 bytes (3×vertex) | 2 bytes attr
        tris = np.frombuffer(raw, dtype=np.uint8).reshape(count, 50)
        verts = np.frombuffer(tris[:, 12:48].tobytes(), dtype=np.float32).reshape(-1, 3)
        faces = np.arange(count * 3, dtype=np.uint32).reshape(-1, 3)
        return verts, faces

    @staticmethod
    def _apply_visual_origin(verts: np.ndarray, xyz, rpy) -> np.ndarray:
        """Bake URDF visual <origin xyz rpy> into vertex positions (R = Rz @ Ry @ Rx)."""
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        R = (Rz @ Ry @ Rx).astype(np.float32)
        return (R @ verts.T).T + np.array(xyz, dtype=np.float32)
