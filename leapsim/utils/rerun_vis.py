from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import rerun as rr
import trimesh
import trimesh.creation


ObjectShape = Tuple[str, Tuple[float, float, float]]
"""(primitive, half_sizes). primitive ∈ {'box', 'ellipsoid'}. Fallback when no mesh is available."""


# Per-link colors for the LEAP hand. Each finger gets a hue; brightness ramps
# base→tip (mcp_joint → pip → dip → fingertip) so you can tell joints apart.
_HAND_LINK_COLORS: dict = {
    "palm_lower":      (140, 140, 140),
    # Index — red
    "mcp_joint":       (140,  40,  40),
    "pip":             (180,  60,  60),
    "dip":             (215,  90,  90),
    "fingertip":       (245, 130, 130),
    # Middle — green
    "mcp_joint_2":     ( 40, 130,  40),
    "pip_2":           ( 60, 170,  60),
    "dip_2":           ( 90, 210,  90),
    "fingertip_2":     (130, 245, 130),
    # Ring — blue
    "mcp_joint_3":     ( 40,  70, 170),
    "pip_3":           ( 60, 110, 210),
    "dip_3":           ( 90, 150, 235),
    "fingertip_3":     (130, 190, 250),
    # Thumb — amber
    "pip_4":           (170, 110,  30),
    "thumb_pip":       (205, 145,  50),
    "thumb_dip":       (230, 180,  80),
    "thumb_fingertip": (250, 215, 110),
}
_OBJECT_COLOR = (220, 200, 100)
_FALLBACK_LINK_COLOR = (200, 200, 200)


def _link_color(name: str) -> Tuple[int, int, int]:
    return _HAND_LINK_COLORS.get(name, _FALLBACK_LINK_COLOR)


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

    def __init__(
        self,
        rr_cfg: dict,
        urdf_path: Path,
        link_names: List[str],
        object_shape: ObjectShape = ('box', (0.04, 0.04, 0.04)),
        observed: Optional[List[dict]] = None,
    ) -> None:
        self._enabled = bool(rr_cfg.get('enabled', False))
        if not self._enabled:
            return

        self._window_length = int(rr_cfg['window_length_steps'])
        self._period        = int(rr_cfg['record_every_n_steps'])
        self._output_dir    = Path(rr_cfg['output_dir'])
        self._fidelity      = rr_cfg.get('fidelity', 'mesh')
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._link_names   = list(link_names)
        self._object_shape = object_shape
        self._meshes       = self._load_meshes(urdf_path) if self._fidelity == 'mesh' else {}

        # Observed envs, one streamed per window (round-robin over the sample).
        # Each carries its own object instance; we preload each object's mesh
        # scaled by that env's actor scale (set_actor_scale is a uniform scale the
        # sim applies that the raw URDF mesh doesn't include). Rendering the wrong
        # object's mesh makes fingers appear to grip mid-air around a floating
        # wrong-shaped object. Falls back to the ObjectShape primitive per slot.
        self._observed = observed or [dict(env_idx=0, object_id=0,
                                           object_type='object', object_urdf=None,
                                           object_scale=1.0)]
        self._obj_meshes: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for s in self._observed:
            mesh = None
            if s.get('object_urdf') is not None:
                mesh = _load_object_mesh(s['object_urdf'])
                if mesh is not None and float(s['object_scale']) != 1.0:
                    verts, faces = mesh
                    mesh = (verts * float(s['object_scale']), faces)
            self._obj_meshes.append(mesh)

        self._active_slot  = 0
        self._global_step  = 0
        self._window_step  = 0
        self._window_count = 0
        self._in_window    = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, frames) -> None:
        if not self._enabled:
            return
        if isinstance(frames, RerunFrame):          # back-compat: accept a single frame
            frames = [frames]

        self._global_step += 1

        if not self._in_window and self._global_step % self._period == 0:
            # Choose which observed env this window streams (round-robin over the
            # sample) BEFORE opening, so the matching object mesh + label render.
            self._active_slot = self._window_count % len(self._observed)
            self._open_window()
            self._in_window   = True
            self._window_step = 0

        if self._in_window:
            frame = frames[self._active_slot]
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
        s = self._observed[self._active_slot]
        path = (
            self._output_dir
            / (f"window_{self._window_count:04d}_step_{self._global_step:08d}"
               f"_env{int(s['env_idx']):04d}_{s['object_type']}.rrd")
        )
        rr.init("leap_hand", recording_id=path.stem, spawn=False)
        rr.save(str(path))

        # Text banner naming the env / object instance this window streams, so the
        # viewer makes clear which of the sampled envs you're looking at.
        rr.log(
            "label",
            rr.TextDocument(
                f"env {int(s['env_idx'])}  |  object: {s['object_type']} "
                f"(id {int(s['object_id'])})  |  scale {float(s['object_scale']):.3f}"
            ),
            static=True,
        )

        for name in self._link_names:
            color = _link_color(name)
            if name in self._meshes:
                verts, faces = self._meshes[name]
                rr.log(
                    f"world/hand/{name}",
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=faces,
                        albedo_factor=color,
                    ),
                    static=True,
                )
            else:
                rr.log(
                    f"world/hand/{name}",
                    rr.Boxes3D(half_sizes=[[0.008, 0.008, 0.008]], colors=[list(color)]),
                    static=True,
                )

        object_mesh = self._obj_meshes[self._active_slot]
        if object_mesh is not None:
            verts, faces = object_mesh
            rr.log(
                "world/object",
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=faces,
                    albedo_factor=_OBJECT_COLOR,
                ),
                static=True,
            )
        else:
            primitive, half_sizes = self._object_shape
            if primitive == 'ellipsoid':
                rr.log(
                    "world/object",
                    rr.Ellipsoids3D(half_sizes=[list(half_sizes)], colors=[list(_OBJECT_COLOR)]),
                    static=True,
                )
            else:
                rr.log(
                    "world/object",
                    rr.Boxes3D(half_sizes=[list(half_sizes)], colors=[list(_OBJECT_COLOR)]),
                    static=True,
                )

    def _log_frame(self, frame: RerunFrame) -> None:
        rr.set_time_sequence("step", self._window_step)

        for i, name in enumerate(self._link_names):
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
    # Hand asset loading (mesh fidelity only)
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


# ------------------------------------------------------------------
# Object mesh loading (module-level, shared across visualizer instances)
# ------------------------------------------------------------------

def _load_object_mesh(urdf_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse object URDF and return a combined (verts, faces) for all visuals.

    Handles:
    - <mesh filename="..."> — loads STL or OBJ via trimesh
    - <box size="x y z">    — tessellated with trimesh
    - <cylinder radius length> — tessellated with trimesh
    - <sphere radius>       — tessellated with trimesh
    """
    try:
        tree = ET.parse(str(urdf_path))
    except ET.ParseError as e:
        print(f"[rerun_vis] failed to parse {urdf_path}: {e}")
        return None

    asset_dir = urdf_path.parent
    parts: list[trimesh.Trimesh] = []

    for link in tree.getroot().findall('link'):
        for visual in link.findall('visual'):
            origin = visual.find('origin')
            if origin is not None:
                xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
            else:
                xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

            geom = visual.find('geometry')
            if geom is None:
                continue

            mesh = _geometry_to_trimesh(geom, asset_dir)
            if mesh is None:
                continue

            # Bake the visual origin transform into vertex positions.
            T = trimesh.transformations.euler_matrix(*rpy)
            T[:3, 3] = xyz
            mesh.apply_transform(T)
            parts.append(mesh)

    if not parts:
        return None

    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    verts = np.array(combined.vertices, dtype=np.float32)
    faces = np.array(combined.faces, dtype=np.uint32)
    return verts, faces


def _geometry_to_trimesh(geom_elem, asset_dir: Path) -> Optional[trimesh.Trimesh]:
    """Convert a URDF <geometry> element to a trimesh.Trimesh."""
    mesh_elem = geom_elem.find('mesh')
    if mesh_elem is not None:
        filename = mesh_elem.get('filename', '')
        scale_attr = mesh_elem.get('scale')
        scale = [float(s) for s in scale_attr.split()] if scale_attr else [1.0, 1.0, 1.0]
        path = asset_dir / filename
        try:
            loaded = trimesh.load(str(path), force='mesh')
        except Exception as e:
            print(f"[rerun_vis] could not load mesh {path}: {e}")
            return None
        loaded.apply_scale(scale)
        return loaded

    box_elem = geom_elem.find('box')
    if box_elem is not None:
        size = [float(v) for v in box_elem.get('size', '0.1 0.1 0.1').split()]
        return trimesh.creation.box(extents=size)

    cyl_elem = geom_elem.find('cylinder')
    if cyl_elem is not None:
        r = float(cyl_elem.get('radius', '0.05'))
        l = float(cyl_elem.get('length', '0.1'))
        return trimesh.creation.cylinder(radius=r, height=l)

    sph_elem = geom_elem.find('sphere')
    if sph_elem is not None:
        r = float(sph_elem.get('radius', '0.05'))
        return trimesh.creation.icosphere(radius=r)

    return None
