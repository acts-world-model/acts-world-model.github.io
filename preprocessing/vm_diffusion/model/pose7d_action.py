"""Convert raw 7D world-frame poses (+ optional normal force) into 9D relative actions.

Input pose format (world frame):
    [x, y, z, qx, qy, qz, qw]        (quaternion order is xyzw)

Output action (9D):
    [dx, dy, dz, r6_1, ..., r6_6]

    first 3 dims: relative translation expressed in the previous sensor frame; when force
                  is enabled, force is encoded as extra virtual displacement along the
                  GelSight normal (local -y by default)
    last 6 dims:  relative rotation R_t^T @ R_{t+1} as Rotation6D
                  (first-two-COLUMNS convention: [col0; col1], NOT identity-subtracted)

Force is never a 10th dimension; it only shifts translation along the normal:
    a_t = [ R_t^T (p_{t+1} - p_t) + k_f * f_n * n_gel_S ,
            Rot6D(R_t^T R_{t+1}) ]

Two action representations share this force operation and differ only in WHICH pose
pair is converted (ACTION_REPRS):

    delta_action     (default) consecutive frames: pair (t, t+1), force f_t
                     a_t = [ R_t^T (p_{t+1} - p_t) + k_f f_t n ,  Rot6D(R_t^T R_{t+1}) ]
                     -> raw_pose_pair_to_force_informed_9d / convert_pose_sequence
    action_chunking  anchor-relative: pair (0, t) with anchor = first frame of the chunk
                     (the training window), force f_t OF THE CURRENT FRAME
                     a_t = [ R_0^T (p_t - p_0) + k_f f_t n ,  Rot6D(R_0^T R_t) ]
                     -> raw_pose_chunk_to_force_informed_9d, materialised per window from
                        the subsampled pose/force streams (subsample_pose_sequence)
                     The normal is NOT rotated: the force operation is identical to
                     delta_action, only the pose pair changes. Chunk rows are not
                     SE(3)-composable (never feed them to compose_action9).

Two force ENCODINGS (FORCE_TERMS) decide which scalar enters that force operation:

    force_term        delta_action row k (pair idx[k]->idx[k+1])   action_chunking row k (pair idx[0]->idx[k+1])
    'absolute'        f[idx[k]]      (start frame; force_agg)       f[idx[k+1]]           (arrival frame)
    'delta'           f[idx[k+1]] - f[idx[k]]                       f[idx[k+1]] - f[idx[0]]

    'absolute' encodes the indentation depth as a per-frame STATE: it does not depend
    on the frame rate (the same 5 N is the same 2.5 mm whether a step is 33 or 167 ms).
    'delta' encodes the CHANGE of indentation over the step -- a rate x duration quantity
    like dp itself, so it grows with the stride. Under 'delta' the two representations
    agree exactly at W == 2 (both f[1]-f[0]), and the delta rows of a window telescope
    to the chunk's last row: sum_k (f[k+1]-f[k]) = f[end]-f[start]. A steady press has
    delta ~ 0: the action then says "pressed deeper / released", not "is pressed".

Config-driven conversion (Pose7DActionConfig) additionally handles fps downsampling:
`target_fps < src_fps` subsamples the POSES by stride and converts between the kept
endpoints. For the pure-pose part this is bit-exact with the ordered SE(3) composition
of the native per-step actions (build_fps6_actions.py convention) -- relative transforms
compose to the endpoint transform. Under 'absolute' the force term is aggregated over the
spanned native steps (`force_agg`: 'mean' or 'start') and applied once per coarse step;
under 'delta' it is the endpoint difference and `force_agg` is ignored.

Run self tests:
    python pose7d_action.py
"""
from dataclasses import dataclass

import numpy as np

# GelSight normal / pressing direction in the sensor frame (local -y).
DEFAULT_NORMAL_AXIS = (0.0, -1.0, 0.0)

# Supported action representations (see module docstring).
ACTION_REPRS = ("delta_action", "action_chunking")

# Gel normal stiffness k: f_n = k * indentation. Measured value 2 N/mm (user-confirmed
# calibration; source reference TBD). The force-to-displacement gain used everywhere
# below is its inverse, k_f = 1/k. NOTE: the parquet column force_<side>_penetration_mm is
# exactly f / 2 (corr 1.000000, |0.5 f - pen| max 0.0000 mm over 43k rows) -- it is this
# same assumption restated upstream, NOT an independent measurement, so it cannot be used
# to validate k.
DEFAULT_STIFFNESS_N_PER_MM = 2.0

# Force encodings (see module docstring): 'absolute' = per-frame indentation state,
# 'delta' = change of indentation over the pair's span.
FORCE_TERMS = ("absolute", "delta")


def stiffness_to_k_f(stiffness_n_per_mm=DEFAULT_STIFFNESS_N_PER_MM):
    """Gel stiffness k [N/mm] -> force-to-displacement gain k_f [m/N].

    displacement = f_n / k, so k_f = 1 / (k * 1000). k = 2 N/mm -> k_f = 0.0005 m/N.
    """
    k = float(stiffness_n_per_mm)
    if k <= 0:
        raise ValueError(f"stiffness must be positive, got {k} N/mm")
    return 1.0 / (k * 1000.0)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"cannot normalize near-zero vector: {v}")
    return v / n


def quaternion_xyzw_to_matrix(q):
    """[qx, qy, qz, qw] (world<-sensor, i.e. R_WS) -> 3x3 rotation matrix."""
    qx, qy, qz, qw = normalize(q)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def matrix_to_rotation_6d(R):
    """3x3 -> 6D, first-two-columns convention: [R[:,0], R[:,1]]."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]])


def rotation_6d_to_matrix(r6):
    """6D -> 3x3 via Gram-Schmidt on the two columns (inverse of matrix_to_rotation_6d)."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[:3], r6[3:6]
    b1 = normalize(a1)
    b2 = normalize(a2 - np.dot(b1, a2) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def compose_action9(actions):
    """Ordered SE(3) composition of (N, 9) pure-pose actions -> one (9,) action.

    Equals the direct action between the endpoint poses. Only valid on actions WITHOUT
    a force term (force displacement would be wrongly rotated by later steps).
    """
    actions = np.asarray(actions, dtype=np.float64)
    p, R = np.zeros(3), np.eye(3)
    for a in actions:
        p = p + R @ a[:3]
        R = R @ rotation_6d_to_matrix(a[3:])
    return np.concatenate([p, matrix_to_rotation_6d(R)])


def raw_pose_pair_to_force_informed_9d(
    pose_t,
    pose_tp1,
    normal_force,
    k_f,
    normal_axis=DEFAULT_NORMAL_AXIS,
):
    """
    pose_t / pose_tp1:
        [x, y, z, qx, qy, qz, qw] in world coordinates
    normal_force:
        scalar normal force in Newtons (force at step t)
    k_f:
        force-to-displacement gain in meters/Newton
    normal_axis:
        GelSight normal in sensor coordinates (default local -y)

    returns:
        (9,) float64 force-informed relative action
    """
    pose_t = np.asarray(pose_t, dtype=np.float64)
    pose_tp1 = np.asarray(pose_tp1, dtype=np.float64)
    if pose_t.shape != (7,) or pose_tp1.shape != (7,):
        raise ValueError(f"poses must be 7D [x,y,z,qx,qy,qz,qw], got {pose_t.shape} / {pose_tp1.shape}")

    p_t_W, q_t = pose_t[:3], pose_t[3:7]
    p_tp1_W, q_tp1 = pose_tp1[:3], pose_tp1[3:7]

    R_t = quaternion_xyzw_to_matrix(q_t)      # R_WS at t
    R_tp1 = quaternion_xyzw_to_matrix(q_tp1)  # R_WS at t+1

    # world displacement -> previous sensor frame
    delta_p_S = R_t.T @ (p_tp1_W - p_t_W)

    # force as virtual displacement along the gel normal (translation only)
    n_gel_S = normalize(normal_axis)
    delta_p_force_informed_S = delta_p_S + k_f * float(normal_force) * n_gel_S

    # relative rotation in the previous sensor frame
    delta_R = R_t.T @ R_tp1
    rot6d = matrix_to_rotation_6d(delta_R)

    return np.concatenate([delta_p_force_informed_S, rot6d])


def raw_pose_sequence_to_force_informed_9d(
    poses,
    normal_forces,
    k_f,
    normal_axis=DEFAULT_NORMAL_AXIS,
):
    """
    poses:         (T, 7) world-frame raw poses
    normal_forces: (T,) or (T-1,) scalar normal forces; force at step t is
                   applied to the action t -> t+1, so only the first T-1 are used
    returns:       (T-1, 9) force-informed actions
    """
    poses = np.asarray(poses, dtype=np.float64)
    normal_forces = np.asarray(normal_forces, dtype=np.float64).reshape(-1)
    T = poses.shape[0]
    if poses.ndim != 2 or poses.shape[1] != 7 or T < 2:
        raise ValueError(f"poses must be (T>=2, 7), got {poses.shape}")
    if normal_forces.shape[0] not in (T, T - 1):
        raise ValueError(f"need {T} or {T - 1} forces for {T} poses, got {normal_forces.shape[0]}")

    return np.stack(
        [
            raw_pose_pair_to_force_informed_9d(poses[t], poses[t + 1], normal_forces[t], k_f, normal_axis)
            for t in range(T - 1)
        ]
    )


def raw_pose_chunk_to_force_informed_9d(
    poses,
    normal_forces,
    k_f,
    normal_axis=DEFAULT_NORMAL_AXIS,
    force_term="absolute",
):
    """`action_chunking`: (W, 7) chunk poses + (W,) per-frame forces -> (W-1, 9) rows.

    Row k (k = 0..W-2) is the force-informed action from the ANCHOR (frame 0) to frame
    k+1, i.e. ``raw_pose_pair_to_force_informed_9d(poses[0], poses[k+1], f_k, ...)`` with

        force_term='absolute'  f_k = normal_forces[k+1]                    (arrival frame)
        force_term='delta'     f_k = normal_forces[k+1] - normal_forces[0] (change since the anchor)

    Why the arrival frame and not "the start frame" like delta_action: every chunk row
    starts at the same anchor, so a start-frame rule would put f[0] on every row and
    carry no information. Under 'absolute' this makes W == 2 differ from the delta_action
    row only by k_f * (f[1] - f[0]) along the normal; under 'delta' the two coincide.

    Rows are relative to one common anchor and are NOT SE(3)-composable.
    """
    if force_term not in FORCE_TERMS:
        raise ValueError(f"force_term must be one of {FORCE_TERMS}, got {force_term!r}")
    poses = np.asarray(poses, dtype=np.float64)
    normal_forces = np.asarray(normal_forces, dtype=np.float64).reshape(-1)
    W = poses.shape[0]
    if poses.ndim != 2 or poses.shape[1] != 7 or W < 2:
        raise ValueError(f"poses must be (W>=2, 7), got {poses.shape}")
    if normal_forces.shape[0] != W:
        raise ValueError(f"need {W} per-frame forces for {W} poses, got {normal_forces.shape[0]}")
    anchor = poses[0]
    f_ref = normal_forces[0] if force_term == "delta" else 0.0
    return np.stack(
        [
            raw_pose_pair_to_force_informed_9d(
                anchor, poses[k + 1], normal_forces[k + 1] - f_ref, k_f, normal_axis)
            for k in range(W - 1)
        ]
    )


# ------------------------------------------------------------------- config-driven pipeline

@dataclass
class Pose7DActionConfig:
    """End-to-end conversion settings: raw pose stream -> training-rate 9D actions.

    use_force            False -> pure pose actions (sensor_actions_fps6.pt convention);
                         True  -> force-informed (needs `normal_forces` at call time)
    stiffness_n_per_mm   gel normal stiffness k in N/mm; the force gain is its inverse,
                         k_f = 1/(k*1000) m/N (only read when use_force)
    k_f                  explicit force-to-displacement gain in m/N; None (default) ->
                         derived from stiffness_n_per_mm, a value overrides it
    normal_axis          GelSight normal in the sensor frame
    src_fps              rate of the raw pose stream (e.g. 30.0)
    target_fps           None / 0 / >= src_fps -> no downsampling; else poses are subsampled
                         by stride = round(src_fps / target_fps) (the AllTracker fps6 stride)
    force_agg            ('absolute' only) force per coarse step: 'mean' over the spanned
                         native steps, or 'start' (the force at the step's first native
                         frame). NOTE the dataclass default is 'mean' while
                         build_force_actions.py forces 'start'; ignored under 'delta'.
    force_term           'absolute' (per-frame indentation state, default) or 'delta'
                         (change of indentation over the pair's span); see module docstring
    action_repr          'delta_action' (consecutive pairs, convert_pose_sequence) or
                         'action_chunking' (anchor-relative rows built per window by
                         raw_pose_chunk_to_force_informed_9d from subsample_pose_sequence)

    target_fps defaults to 15 (the stride-2 Wan2.2 tree is the default since 2026-09-07);
    the builders always pass it explicitly.
    """
    use_force: bool = True
    stiffness_n_per_mm: float = DEFAULT_STIFFNESS_N_PER_MM
    k_f: float | None = None
    normal_axis: tuple = DEFAULT_NORMAL_AXIS
    src_fps: float = 30.0
    target_fps: float | None = 15.0
    force_agg: str = "mean"
    force_term: str = "absolute"
    action_repr: str = "delta_action"

    def __post_init__(self):
        if self.action_repr not in ACTION_REPRS:
            raise ValueError(f"action_repr must be one of {ACTION_REPRS}, got {self.action_repr!r}")
        if self.force_term not in FORCE_TERMS:
            raise ValueError(f"force_term must be one of {FORCE_TERMS}, got {self.force_term!r}")

    def resolved_k_f(self) -> float:
        return self.k_f if self.k_f is not None else stiffness_to_k_f(self.stiffness_n_per_mm)

    def stride(self) -> int:
        if not self.target_fps or self.target_fps >= self.src_fps:
            return 1
        return max(1, int(round(self.src_fps / self.target_fps)))


def convert_pose_sequence(poses, normal_forces=None, config: Pose7DActionConfig = None):
    """(T, 7) raw poses [+ (T,) or (T-1,) native-rate forces] -> downsampled 9D actions.

    Downsampling subsamples the poses (frame_idx = arange(0, T, stride)) and converts
    between kept endpoints -- for the pose part identical to composing the native
    per-step actions, without compounding error. Force for coarse step k: under
    force_term='absolute' it aggregates the native forces spanned by
    [frame_idx[k], frame_idx[k+1]) (`force_agg`); under 'delta' it is the endpoint
    difference f[frame_idx[k+1]] - f[frame_idx[k]] (needs all T per-frame forces).

    This is the `delta_action` representation; for `action_chunking` store the streams
    from ``subsample_pose_sequence`` and build rows per window with
    ``raw_pose_chunk_to_force_informed_9d``.

    returns:
        actions   (F-1, 9) float64
        frame_idx (F,)     int64   indices into the native pose stream; action[k]
                                   covers native frames [frame_idx[k], frame_idx[k+1])
    """
    if config is None:
        config = Pose7DActionConfig()
    if config.action_repr != "delta_action":
        raise ValueError(
            f"convert_pose_sequence builds delta_action rows only; config.action_repr="
            f"{config.action_repr!r} -> use subsample_pose_sequence + raw_pose_chunk_to_force_informed_9d"
        )
    poses = np.asarray(poses, dtype=np.float64)
    T = poses.shape[0]
    if poses.ndim != 2 or poses.shape[1] != 7 or T < 2:
        raise ValueError(f"poses must be (T>=2, 7), got {poses.shape}")

    if config.use_force:
        if normal_forces is None:
            raise ValueError("config.use_force=True but normal_forces is None")
        normal_forces = np.asarray(normal_forces, dtype=np.float64).reshape(-1)
        if normal_forces.shape[0] not in (T, T - 1):
            raise ValueError(f"need {T} or {T - 1} forces for {T} poses, got {normal_forces.shape[0]}")
        if config.force_term == "delta" and normal_forces.shape[0] != T:
            raise ValueError(f"force_term='delta' needs one force per frame ({T}), "
                             f"got {normal_forces.shape[0]}")
        step_forces = normal_forces[: T - 1]  # force of native step t -> t+1

    frame_idx = np.arange(0, T, config.stride(), dtype=np.int64)
    if frame_idx.shape[0] < 2:
        raise ValueError(f"stride {config.stride()} leaves <2 frames of {T}")

    k_f = config.resolved_k_f() if config.use_force else 0.0
    actions = []
    for k in range(frame_idx.shape[0] - 1):
        s, e = int(frame_idx[k]), int(frame_idx[k + 1])
        if not config.use_force:
            f = 0.0
        elif config.force_term == "delta":
            f = float(normal_forces[e] - normal_forces[s])
        else:
            span = step_forces[s:e]
            f = float(span.mean()) if config.force_agg == "mean" else float(span[0])
        actions.append(
            raw_pose_pair_to_force_informed_9d(poses[s], poses[e], f, k_f, config.normal_axis)
        )
    return np.stack(actions), frame_idx


def subsample_pose_sequence(poses, normal_forces, config: Pose7DActionConfig = None):
    """(T, 7) raw poses + (T,) native-rate forces -> per-frame streams at the target rate.

    The `action_chunking` counterpart of ``convert_pose_sequence``: the same frame grid
    ``frame_idx = arange(0, T, stride)`` is kept, but poses and forces are stored PER
    FRAME (not converted), because the anchor-relative rows depend on the window they
    are built for. Force is point-sampled at the native frame ``frame_idx[k]``
    (``force_sample='frame'``) -- the same value as force_fps6.pt[k] and as the
    delta_action file's 'start' force at stride 5.

    returns:
        poses_sub  (F, 7) float64
        forces_sub (F,)   float64
        frame_idx  (F,)   int64
    """
    if config is None:
        config = Pose7DActionConfig(action_repr="action_chunking")
    poses = np.asarray(poses, dtype=np.float64)
    normal_forces = np.asarray(normal_forces, dtype=np.float64).reshape(-1)
    T = poses.shape[0]
    if poses.ndim != 2 or poses.shape[1] != 7 or T < 2:
        raise ValueError(f"poses must be (T>=2, 7), got {poses.shape}")
    if normal_forces.shape[0] != T:
        raise ValueError(f"need {T} per-frame forces for {T} poses, got {normal_forces.shape[0]}")
    frame_idx = np.arange(0, T, config.stride(), dtype=np.int64)
    if frame_idx.shape[0] < 2:
        raise ValueError(f"stride {config.stride()} leaves <2 frames of {T}")
    return poses[frame_idx].copy(), normal_forces[frame_idx].copy(), frame_idx


# ----------------------------------------------------------------------------- tests

def _quat_z(deg):
    h = np.deg2rad(deg) / 2.0
    return [0.0, 0.0, np.sin(h), np.cos(h)]


def _random_poses(rng, T):
    return np.stack(
        [np.concatenate([rng.standard_normal(3) * 0.1, normalize(rng.standard_normal(4))]) for _ in range(T)]
    )


def _test_zero_motion_zero_force():
    pose = np.array([0.1, 0.2, 0.3] + _quat_z(30.0))
    a = raw_pose_pair_to_force_informed_9d(pose, pose, normal_force=0.0, k_f=0.002)
    assert np.allclose(a[:3], 0.0, atol=1e-12), a[:3]
    assert np.allclose(rotation_6d_to_matrix(a[3:]), np.eye(3), atol=1e-12)


def _test_zero_motion_positive_force():
    pose = np.array([0.1, 0.2, 0.3] + _quat_z(30.0))
    f_n, k_f = 4.0, 0.002
    a = raw_pose_pair_to_force_informed_9d(pose, pose, f_n, k_f)
    assert np.allclose(a[:3], [0.0, -k_f * f_n, 0.0], atol=1e-12), a[:3]
    # more force -> more displacement along the pressing direction (-y)
    a2 = raw_pose_pair_to_force_informed_9d(pose, pose, 2 * f_n, k_f)
    assert a2[1] < a[1] < 0.0


def _test_spec_numerical_example():
    pose_t = np.array([0.30, 0.20, 0.10] + _quat_z(45.0))
    pose_tp1 = np.array([0.31, 0.20, 0.10] + _quat_z(90.0))
    a = raw_pose_pair_to_force_informed_9d(pose_t, pose_tp1, normal_force=4.0, k_f=0.002)
    s = np.sqrt(0.5)
    expected = np.array([0.01 * s, -0.01 * s - 0.008, 0.0, s, s, 0.0, -s, s, 0.0])
    assert np.allclose(a, expected, atol=1e-4), f"\ngot      {a}\nexpected {expected}"


def _test_translation_in_previous_sensor_frame():
    # sensor rotated 90deg about world z; world +x displacement must appear as
    # sensor-frame [0, -d, 0] (world x is the sensor's -y axis)
    pose_t = np.array([0.0, 0.0, 0.0] + _quat_z(90.0))
    pose_tp1 = np.array([0.05, 0.0, 0.0] + _quat_z(90.0))
    a = raw_pose_pair_to_force_informed_9d(pose_t, pose_tp1, 0.0, k_f=0.002)
    assert np.allclose(a[:3], [0.0, -0.05, 0.0], atol=1e-12), a[:3]
    assert np.allclose(rotation_6d_to_matrix(a[3:]), np.eye(3), atol=1e-12)


def _test_rot6d_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = normalize(rng.standard_normal(4))
        R = quaternion_xyzw_to_matrix(q)
        R_rec = rotation_6d_to_matrix(matrix_to_rotation_6d(R))
        assert np.allclose(R_rec, R, atol=1e-10)


def _test_random_consistency():
    # reconstruct pose_tp1 from pose_t + action (with force removed) and compare
    rng = np.random.default_rng(1)
    f_n, k_f = 3.0, 0.002
    n_gel_S = np.array(DEFAULT_NORMAL_AXIS)
    for _ in range(50):
        p_t, p_tp1 = rng.standard_normal(3) * 0.1, rng.standard_normal(3) * 0.1
        q_t, q_tp1 = normalize(rng.standard_normal(4)), normalize(rng.standard_normal(4))
        a = raw_pose_pair_to_force_informed_9d(
            np.concatenate([p_t, q_t]), np.concatenate([p_tp1, q_tp1]), f_n, k_f
        )
        R_t = quaternion_xyzw_to_matrix(q_t)
        delta_p_S = a[:3] - k_f * f_n * n_gel_S
        assert np.allclose(p_t + R_t @ delta_p_S, p_tp1, atol=1e-10)
        assert np.allclose(R_t @ rotation_6d_to_matrix(a[3:]), quaternion_xyzw_to_matrix(q_tp1), atol=1e-10)


def _test_unnormalized_quaternion():
    pose_t = np.array([0.0, 0.0, 0.0] + list(2.0 * np.array(_quat_z(45.0))))
    pose_ref = np.array([0.0, 0.0, 0.0] + _quat_z(45.0))
    pose_tp1 = np.array([0.01, 0.0, 0.0] + _quat_z(90.0))
    a = raw_pose_pair_to_force_informed_9d(pose_t, pose_tp1, 1.0, k_f=0.001)
    a_ref = raw_pose_pair_to_force_informed_9d(pose_ref, pose_tp1, 1.0, k_f=0.001)
    assert np.allclose(a, a_ref, atol=1e-12)


def _test_sequence():
    poses = np.stack(
        [
            np.array([0.30, 0.20, 0.10] + _quat_z(45.0)),
            np.array([0.31, 0.20, 0.10] + _quat_z(90.0)),
            np.array([0.31, 0.20, 0.10] + _quat_z(90.0)),
        ]
    )
    forces = np.array([4.0, 0.0, 5.0])  # last force unused (T forces for T poses)
    actions = raw_pose_sequence_to_force_informed_9d(poses, forces, k_f=0.002)
    assert actions.shape == (2, 9)
    a0 = raw_pose_pair_to_force_informed_9d(poses[0], poses[1], 4.0, 0.002)
    assert np.allclose(actions[0], a0)
    assert np.allclose(actions[1], np.concatenate([[0, 0, 0], matrix_to_rotation_6d(np.eye(3))]))
    # (T-1,) forces also accepted
    actions2 = raw_pose_sequence_to_force_informed_9d(poses, forces[:2], k_f=0.002)
    assert np.allclose(actions, actions2)


def _test_config_no_force():
    # use_force=False ignores forces entirely and matches f=0 pairwise conversion
    rng = np.random.default_rng(2)
    poses = _random_poses(rng, 4)
    cfg = Pose7DActionConfig(use_force=False, src_fps=30.0, target_fps=None)
    actions, frame_idx = convert_pose_sequence(poses, None, cfg)
    assert actions.shape == (3, 9) and np.array_equal(frame_idx, np.arange(4))
    ref = raw_pose_sequence_to_force_informed_9d(poses, np.zeros(4), k_f=0.0)
    assert np.allclose(actions, ref)


def _test_config_no_downsample_when_target_ge_src():
    rng = np.random.default_rng(3)
    poses = _random_poses(rng, 5)
    forces = rng.uniform(0, 5, 5)
    for tfps in (None, 0, 30.0, 60.0):
        cfg = Pose7DActionConfig(use_force=True, k_f=0.002, src_fps=30.0, target_fps=tfps)
        actions, frame_idx = convert_pose_sequence(poses, forces, cfg)
        assert cfg.stride() == 1 and actions.shape == (4, 9)
        assert np.allclose(actions, raw_pose_sequence_to_force_informed_9d(poses, forces, 0.002))


def _test_config_downsample_matches_composition():
    # pure-pose downsampled action == ordered composition of the native per-step actions
    # (the build_fps6_actions.py invariant), and frame_idx matches the fps6 stride
    rng = np.random.default_rng(4)
    poses = _random_poses(rng, 16)
    cfg = Pose7DActionConfig(use_force=False, src_fps=30.0, target_fps=6.0)
    assert cfg.stride() == 5
    actions, frame_idx = convert_pose_sequence(poses, None, cfg)
    assert np.array_equal(frame_idx, [0, 5, 10, 15]) and actions.shape == (3, 9)
    native = raw_pose_sequence_to_force_informed_9d(poses, np.zeros(16), k_f=0.0)
    for k in range(3):
        composed = compose_action9(native[5 * k: 5 * (k + 1)])
        assert np.allclose(actions[k], composed, atol=1e-10)


def _test_config_downsample_force_agg():
    rng = np.random.default_rng(5)
    poses = _random_poses(rng, 11)
    forces = rng.uniform(0, 6, 10)  # (T-1,) native step forces
    k_f = 0.002
    for agg in ("mean", "start"):
        cfg = Pose7DActionConfig(use_force=True, k_f=k_f, src_fps=30.0, target_fps=6.0, force_agg=agg)
        actions, frame_idx = convert_pose_sequence(poses, forces, cfg)
        assert np.array_equal(frame_idx, [0, 5, 10]) and actions.shape == (2, 9)
        no_force, _ = convert_pose_sequence(
            poses, None, Pose7DActionConfig(use_force=False, src_fps=30.0, target_fps=6.0))
        for k, (s, e) in enumerate(((0, 5), (5, 10))):
            f = forces[s:e].mean() if agg == "mean" else forces[s]
            expected_dy = no_force[k, 1] - k_f * f
            assert np.isclose(actions[k, 1], expected_dy, atol=1e-12)
            assert np.allclose(actions[k, [0, 2]], no_force[k, [0, 2]], atol=1e-12)  # only -y shifts
            assert np.allclose(actions[k, 3:], no_force[k, 3:], atol=1e-12)          # rotation untouched


def _test_stiffness_parameter():
    # k = 2 N/mm -> k_f = 0.0005 m/N; 4 N with zero motion -> 2 mm along -y
    assert np.isclose(stiffness_to_k_f(2.0), 0.0005)
    pose = np.array([0.1, 0.2, 0.3] + _quat_z(30.0))
    poses = np.stack([pose, pose])
    cfg = Pose7DActionConfig(use_force=True, src_fps=30.0, target_fps=None)  # defaults: k=2 N/mm
    assert np.isclose(cfg.resolved_k_f(), 0.0005)
    actions, _ = convert_pose_sequence(poses, np.array([4.0]), cfg)
    assert np.allclose(actions[0, :3], [0.0, -0.002, 0.0], atol=1e-12), actions[0, :3]
    # explicit k_f overrides the stiffness-derived gain
    cfg_kf = Pose7DActionConfig(use_force=True, k_f=0.002, src_fps=30.0, target_fps=None)
    assert np.isclose(cfg_kf.resolved_k_f(), 0.002)
    actions_kf, _ = convert_pose_sequence(poses, np.array([4.0]), cfg_kf)
    assert np.allclose(actions_kf[0, :3], [0.0, -0.008, 0.0], atol=1e-12)
    # a softer gel (smaller k) presses further for the same force
    cfg_soft = Pose7DActionConfig(use_force=True, stiffness_n_per_mm=1.0, src_fps=30.0, target_fps=None)
    actions_soft, _ = convert_pose_sequence(poses, np.array([4.0]), cfg_soft)
    assert actions_soft[0, 1] < actions[0, 1] < 0.0


def _test_config_force_required():
    poses = _random_poses(np.random.default_rng(6), 3)
    try:
        convert_pose_sequence(poses, None, Pose7DActionConfig(use_force=True))
    except ValueError:
        return
    raise AssertionError("use_force=True with no forces should raise")


def _test_chunk_w2_equals_delta():
    # at W=2 the chunk row is the pair action with the END frame's force
    rng = np.random.default_rng(7)
    poses = _random_poses(rng, 2)
    forces = np.array([1.5, 4.0])
    k_f = 0.002
    chunk = raw_pose_chunk_to_force_informed_9d(poses, forces, k_f)
    assert chunk.shape == (1, 9)
    assert np.allclose(chunk[0], raw_pose_pair_to_force_informed_9d(poses[0], poses[1], forces[1], k_f), atol=1e-12)
    # with zero force the two representations coincide exactly
    chunk0 = raw_pose_chunk_to_force_informed_9d(poses, np.zeros(2), k_f)
    delta0 = raw_pose_sequence_to_force_informed_9d(poses, np.zeros(2), k_f)
    assert np.allclose(chunk0, delta0, atol=1e-12)


def _test_chunk_zero_motion_force_only():
    pose = np.array([0.1, 0.2, 0.3] + _quat_z(30.0))
    poses = np.stack([pose, pose, pose])
    forces = np.array([0.0, 4.0, 2.0])
    k_f = 0.002
    rows = raw_pose_chunk_to_force_informed_9d(poses, forces, k_f)
    assert rows.shape == (2, 9)
    for k in range(2):
        assert np.allclose(rows[k, :3], [0.0, -k_f * forces[k + 1], 0.0], atol=1e-12), rows[k, :3]
        assert np.allclose(rotation_6d_to_matrix(rows[k, 3:]), np.eye(3), atol=1e-12)
    # the anchor's own force (index 0) is never used
    rows2 = raw_pose_chunk_to_force_informed_9d(poses, np.array([9.0, 4.0, 2.0]), k_f)
    assert np.allclose(rows, rows2, atol=1e-12)


def _test_chunk_reconstruction():
    # every row reconstructs frame k+1 from the anchor once its force term is removed
    rng = np.random.default_rng(8)
    W, k_f = 16, 0.002
    poses = _random_poses(rng, W)
    forces = rng.uniform(0, 6, W)
    rows = raw_pose_chunk_to_force_informed_9d(poses, forces, k_f)
    assert rows.shape == (W - 1, 9)
    n = np.array(DEFAULT_NORMAL_AXIS)
    R_0 = quaternion_xyzw_to_matrix(poses[0, 3:])
    for k in range(W - 1):
        dp = rows[k, :3] - k_f * forces[k + 1] * n
        assert np.allclose(poses[0, :3] + R_0 @ dp, poses[k + 1, :3], atol=1e-10)
        assert np.allclose(R_0 @ rotation_6d_to_matrix(rows[k, 3:]),
                           quaternion_xyzw_to_matrix(poses[k + 1, 3:]), atol=1e-10)


def _test_chunk_not_composable():
    # anchor-relative rows are not per-step actions: composing them is meaningless
    rng = np.random.default_rng(9)
    poses = _random_poses(rng, 3)
    rows = raw_pose_chunk_to_force_informed_9d(poses, np.array([0.0, 3.0, 3.0]), 0.002)
    assert not np.allclose(compose_action9(rows[:2]), rows[1], atol=1e-6)
    # wrong force length is rejected
    try:
        raw_pose_chunk_to_force_informed_9d(poses, np.zeros(2), 0.002)
    except ValueError:
        pass
    else:
        raise AssertionError("(W-1,) forces should be rejected for chunk rows")


def _test_subsample_pose_sequence():
    rng = np.random.default_rng(10)
    poses = _random_poses(rng, 16)
    forces = rng.uniform(0, 6, 16)
    cfg = Pose7DActionConfig(use_force=True, k_f=0.002, src_fps=30.0, target_fps=6.0,
                             action_repr="action_chunking")
    poses_sub, forces_sub, frame_idx = subsample_pose_sequence(poses, forces, cfg)
    assert np.array_equal(frame_idx, [0, 5, 10, 15])
    assert np.allclose(poses_sub, poses[[0, 5, 10, 15]]) and np.allclose(forces_sub, forces[[0, 5, 10, 15]])
    # pose part of a W=2 chunk row over the subsampled grid == the delta_action row at stride
    delta, _ = convert_pose_sequence(
        poses, forces, Pose7DActionConfig(use_force=True, k_f=0.002, src_fps=30.0, target_fps=6.0, force_agg="start"))
    chunk = raw_pose_chunk_to_force_informed_9d(poses_sub[:2], forces_sub[:2], 0.002)
    n = np.array(DEFAULT_NORMAL_AXIS)
    assert np.allclose(chunk[0, 3:], delta[0, 3:], atol=1e-12)
    assert np.allclose(chunk[0, :3] - 0.002 * forces_sub[1] * n, delta[0, :3] - 0.002 * forces_sub[0] * n, atol=1e-12)
    # convert_pose_sequence refuses a chunking config
    try:
        convert_pose_sequence(poses, forces, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("convert_pose_sequence should reject action_repr=action_chunking")
    try:
        Pose7DActionConfig(action_repr="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown action_repr should be rejected")


def _test_force_term_delta_pair_rows():
    # delta_action under force_term='delta': force scalar is f[e] - f[s]; pose part unchanged
    rng = np.random.default_rng(11)
    poses = _random_poses(rng, 11)
    forces = rng.uniform(0, 6, 11)
    k_f = 0.002
    n = np.array(DEFAULT_NORMAL_AXIS)
    for tfps, grid in ((None, np.arange(11)), (6.0, np.array([0, 5, 10]))):
        cfg_abs = Pose7DActionConfig(use_force=True, k_f=k_f, src_fps=30.0, target_fps=tfps,
                                     force_agg="start", force_term="absolute")
        cfg_dlt = Pose7DActionConfig(use_force=True, k_f=k_f, src_fps=30.0, target_fps=tfps,
                                     force_term="delta")
        a_abs, fi = convert_pose_sequence(poses, forces, cfg_abs)
        a_dlt, fi2 = convert_pose_sequence(poses, forces, cfg_dlt)
        assert np.array_equal(fi, grid) and np.array_equal(fi2, grid)
        for k in range(len(grid) - 1):
            s, e = grid[k], grid[k + 1]
            assert np.allclose(a_dlt[k, 3:], a_abs[k, 3:], atol=1e-12)
            assert np.allclose(a_dlt[k, :3] - k_f * (forces[e] - forces[s]) * n,
                               a_abs[k, :3] - k_f * forces[s] * n, atol=1e-12)
    # 'delta' needs one force per frame
    try:
        convert_pose_sequence(poses, forces[:-1], Pose7DActionConfig(use_force=True, k_f=k_f,
                                                                     src_fps=30.0, target_fps=None,
                                                                     force_term="delta"))
    except ValueError:
        pass
    else:
        raise AssertionError("force_term='delta' with T-1 forces should raise")
    try:
        Pose7DActionConfig(force_term="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown force_term should be rejected")


def _test_force_term_delta_w2_and_telescoping():
    # under 'delta' the two representations coincide at W == 2, and the window's delta
    # rows telescope to the chunk's last row's force
    rng = np.random.default_rng(12)
    W, k_f = 16, 0.002
    poses = _random_poses(rng, W)
    forces = rng.uniform(0, 6, W)
    n = np.array(DEFAULT_NORMAL_AXIS)
    delta = convert_pose_sequence(poses, forces, Pose7DActionConfig(
        use_force=True, k_f=k_f, src_fps=30.0, target_fps=None, force_term="delta"))[0]
    chunk = raw_pose_chunk_to_force_informed_9d(poses, forces, k_f, force_term="delta")
    assert np.allclose(chunk[0], delta[0], atol=1e-12)
    assert np.allclose(raw_pose_chunk_to_force_informed_9d(poses[:2], forces[:2], k_f, force_term="delta")[0],
                       delta[0], atol=1e-12)
    # chunk row k == compose(pure delta rows 0..k) + k_f * (f[k+1] - f[0]) * n
    pure = delta.copy()
    pure[:, :3] -= k_f * (forces[1:] - forces[:-1])[:, None] * n
    for k in range(W - 1):
        c = compose_action9(pure[: k + 1])
        assert np.allclose(chunk[k, 3:], c[3:], atol=1e-9)
        assert np.allclose(chunk[k, :3], c[:3] + k_f * (forces[k + 1] - forces[0]) * n, atol=1e-9)
    # telescoping of the force scalars themselves
    assert np.isclose(np.sum(forces[1:] - forces[:-1]), forces[-1] - forces[0])
    # 'absolute' chunk rows are unchanged by the new argument's default
    assert np.allclose(raw_pose_chunk_to_force_informed_9d(poses, forces, k_f),
                       raw_pose_chunk_to_force_informed_9d(poses, forces, k_f, force_term="absolute"))


def _test_force_term_absolute_regression():
    # the default encoding is bit-identical to the pre-force_term implementation
    rng = np.random.default_rng(13)
    poses = _random_poses(rng, 21)
    forces = rng.uniform(0, 6, 21)
    k_f = 0.0005
    n = np.array(DEFAULT_NORMAL_AXIS)
    for agg in ("start", "mean"):
        cfg = Pose7DActionConfig(use_force=True, k_f=k_f, src_fps=30.0, target_fps=6.0, force_agg=agg)
        assert cfg.force_term == "absolute"
        a, fi = convert_pose_sequence(poses, forces, cfg)
        for k in range(len(fi) - 1):
            s, e = fi[k], fi[k + 1]
            f = forces[s:e].mean() if agg == "mean" else forces[s]
            ref = raw_pose_pair_to_force_informed_9d(poses[s], poses[e], f, k_f)
            assert np.array_equal(a[k], ref)
    ch = raw_pose_chunk_to_force_informed_9d(poses[:5], forces[:5], k_f)
    for k in range(4):
        assert np.array_equal(ch[k], raw_pose_pair_to_force_informed_9d(poses[0], poses[k + 1], forces[k + 1], k_f))


if __name__ == "__main__":
    tests = [
        _test_zero_motion_zero_force,
        _test_zero_motion_positive_force,
        _test_spec_numerical_example,
        _test_translation_in_previous_sensor_frame,
        _test_rot6d_round_trip,
        _test_random_consistency,
        _test_unnormalized_quaternion,
        _test_sequence,
        _test_config_no_force,
        _test_config_no_downsample_when_target_ge_src,
        _test_config_downsample_matches_composition,
        _test_config_downsample_force_agg,
        _test_stiffness_parameter,
        _test_config_force_required,
        _test_chunk_w2_equals_delta,
        _test_chunk_zero_motion_force_only,
        _test_chunk_reconstruction,
        _test_chunk_not_composable,
        _test_subsample_pose_sequence,
        _test_force_term_delta_pair_rows,
        _test_force_term_delta_w2_and_telescoping,
        _test_force_term_absolute_regression,
    ]
    for t in tests:
        t()
        print(f"[pass] {t.__name__}")
    print(f"all {len(tests)} tests passed")
