"""Prompt context for the G1 standing LOKA loop."""

from __future__ import annotations

from pathlib import Path

import mujoco

from loka.control.locomotion import LocomotionController
from loka.agent.robot_context import build_robot_model_context


def default_stand_objective() -> str:
    return (
        "Hold a robust standing balance on the Unitree G1, and accept operator "
        "requests to crouch, stand tall, lean, yaw, or walk. On walk requests, "
        "set gait.mode/speed/heading (and optionally cadence / swing height). "
        "On faults, queue Belief_Searches so the model matches the plant, and set "
        "Task_Targets for the directive. Do not generate code."
    )


#: Gait targets the compact prompt shows; the rest stay at their schedule values.
_COMPACT_GAIT_KEYS = (
    "gait.mode",
    "gait.speed",
    "gait.heading",
    "gait.goal_active",
    "gait.goal_x",
    "gait.goal_y",
    "gait.goal_tolerance",
    "gait.swing_height",
)


def format_stand_configuration(controller: LocomotionController, *, compact: bool = False) -> str:
    """Live MPC costs and DCM gait knobs, current values only.

    ``compact`` drops the MPC costs (the orchestrator does not retune them) and
    keeps the navigation-relevant gait targets on one line.
    """
    tasks = controller.task_snapshot()
    from loka.control.gait import gait_mode_name

    if compact:
        bits = []
        for key in _COMPACT_GAIT_KEYS:
            if key not in tasks:
                continue
            value = float(tasks[key])
            shown = gait_mode_name(value) if key == "gait.mode" else f"{value:.3g}"
            bits.append(f"{key}={shown}")
        return "## GAIT TARGETS\n  " + " ".join(bits)
    weights = controller.tunables()

    lines = ["## CURRENT MPC TUNABLES"]
    mpc_paths = [path for path in sorted(weights) if path.startswith("mpc.")]
    if not mpc_paths:
        lines.append("  (none)")
    for path in mpc_paths:
        lines.append(f"  {path}: {weights[path]:.6g}")
    lines.append("## CURRENT DCM TUNABLES")
    gait_keys = [key for key in sorted(tasks) if str(key).startswith("gait.")]
    if not gait_keys:
        lines.append("  (none)")
    for key in gait_keys:
        value = float(tasks[key])
        if key == "gait.mode":
            lines.append(f"  {key}: {value:.4g} ({gait_mode_name(value)})")
        else:
            lines.append(f"  {key}: {value:.6g}")
    return "\n".join(lines)


def format_compact_vision_block(spec, nominal_pose, floor_cell: float | None) -> str:
    """The camera facts a planner needs, without the derivations.

    Camera facts only: what is in the scene is for the model to find in the image.
    """
    import math

    h = float(nominal_pose.xyz[2])
    p = spec.pitch_rad
    horizon = spec.cy - spec.fy * math.tan(p)
    ranges = []
    for v in (int(math.ceil(horizon)) + 15, 256, spec.height - 1):
        yn = (v - spec.cy) / spec.fy
        denom = math.sin(p) + yn * math.cos(p)
        if denom > 1e-6 and v < spec.height:
            ranges.append(f"row {v}: {h * (math.cos(p) - yn * math.sin(p)) / denom:.1f} m")
    floor = f" The floor is a checkerboard of {floor_cell:.2f} m squares." if floor_cell else ""
    return "\n".join(
        [
            "## CAMERA",
            f"One {spec.width}x{spec.height} RGB head-camera image per message, pitched "
            f"{spec.pitch_down_deg:.0f} deg down, horizontal FOV {spec.fovx_deg:.0f} deg, "
            f"about {h:.2f} m above the floor. The image's left is the robot's left. Horizon "
            f"near row {horizon:.0f}; floor distance ahead by image row: {', '.join(ranges)}.{floor}",
            "The plant does not see obstacles: route around them with waypoints. Stairs and "
            "curbs are terrain: declare them with a stairs Model_Mutation and set the goal on top.",
        ]
    )


def format_vision_block(spec, nominal_pose, floor_cell: float | None) -> str:
    """System-prompt block describing the head camera and how to use it."""
    import math

    h = float(nominal_pose.xyz[2])
    p = spec.pitch_rad
    horizon = spec.cy - spec.fy * math.tan(p)
    ranges = []
    for v in (int(math.ceil(horizon)) + 15, 100, 150, 200, 256, 300, 350, 400, 450, spec.height - 1):
        yn = (v - spec.cy) / spec.fy
        denom = math.sin(p) + yn * math.cos(p)
        if denom <= 1e-6 or v >= spec.height:
            continue
        forward = h * (math.cos(p) - yn * math.sin(p)) / denom
        half_width = h * ((spec.width - spec.cx) / spec.fx) / denom
        ranges.append(f"  row {v:3d}: floor {forward:5.2f} m ahead, image spans +/-{half_width:4.2f} m sideways")
    floor = (
        f"The floor is a blue checkerboard of {floor_cell:.2f} m squares, alternating "
        f"light and dark, with thin white lines every {2 * floor_cell:.2f} m. Count "
        "squares to measure distances and sizes on the floor."
        if floor_cell
        else "The floor has no metric pattern."
    )
    lines = [
        "## EGOCENTRIC CAMERA",
        "Each user message carries one RGB image from ego_cam, a camera on the front of "
        "the G1 head, rendered at the instant the message was sent. It shows what the "
        "robot would see. Use it to find obstacles, free space, and objects on the way "
        "to the goal. Telemetry remains the source of truth for the robot's own state.",
        "",
        "### Camera and image",
        f"- RGB only (no depth), {spec.width} x {spec.height} px, noiseless render, no lens distortion.",
        f"- Pinhole intrinsics: fx = fy = {spec.fx:.1f} px, cx = {spec.cx:.1f}, cy = {spec.cy:.1f}. "
        f"Vertical FOV {spec.fovy_deg:.1f} deg, horizontal FOV {spec.fovx_deg:.1f} deg.",
        f"- Mounted on torso_link at {tuple(round(v, 3) for v in _ego_mount())} m "
        "(x forward, z up), midway between the head's stereo cameras; the head is "
        "fixed to the torso, so it turns with the waist and torso.",
        f"- The optical axis points along the torso's forward direction, pitched "
        f"{spec.pitch_down_deg:.1f} deg below horizontal.",
        f"- In the nominal stand the camera is {h:.3f} m above the floor. Its live height, "
        "heading, pitch, and roll at capture are in each message's CAMERA FRAME section; "
        "a crouch or lean changes them.",
        "- Pixel (u, v): u grows to the right, v grows downward, (0, 0) is the top-left corner. "
        "The image's left half is the robot's left (+y in its heading frame).",
        f"- With the nominal pitch the horizon is row {horizon:.0f}. Everything below it is "
        "floor or objects standing on it; the sky gradient fills the top rows.",
        "- The robot's own arms or hands can appear at the bottom edge; they are not obstacles.",
        "- Lighting is a directional overhead light plus a headlight, so objects cast short "
        "shadows straight down.",
        f"- {floor}",
        "",
        "### From a pixel to the floor",
        "Use the camera height h, pitch p, and yaw from CAMERA FRAME. For the pixel where an "
        "object meets the floor (its lowest visible point):",
        "  xn = (u - cx) / fx,  yn = (v - cy) / fy,  d = sin(p) + yn cos(p)  (d <= 0: above the horizon)",
        "  ahead X = h (cos(p) - yn sin(p)) / d,   left Y = -h xn / d   (camera heading frame, m)",
        "  world x = cam_x + X cos(yaw) - Y sin(yaw),   world y = cam_y + X sin(yaw) + Y cos(yaw)",
        "Bearing of a column, ignoring pitch: atan((cx - u) / fx), positive to the left. An object "
        f"of width W at slant range R spans about fx W / R pixels.",
        "Reference ranges for the centre column in the nominal stand:",
        *ranges,
        "Image ranges are good to about 10% close in and degrade with distance. Re-check from a "
        "nearer frame before committing to a tight gap.",
        "",
        "### Navigating with the image",
        "- Consult the image when the path matters: before starting a walk, near listeners, and "
        "whenever the goal is not yet reached. If the image shows nothing in the way, proceed "
        "as you would without it.",
        "- The plant walks a straight line to gait.goal_x/goal_y and does not see obstacles. To "
        "go around one, set an intermediate waypoint as the goal, at least 0.5 m clear of the "
        "obstacle's edge (the robot is about 0.5 m wide), and arm a near_xy listener there; "
        "then set the next waypoint or the final goal. Keep the locked directive's goal as the "
        "end point.",
        "- Put a detour waypoint beside the obstacle (level with it or past its near edge), not "
        "straight to the robot's side: e.g. robot at (0,0) facing +x, an obstacle centred at (1.5,0), "
        "goal (3,0) -> waypoints (1.0,-0.9), (2.0,-0.9), then (3,0), each a turn of about 42 deg, "
        "rather than (0,-0.9), a 90 deg turn. Keep each turn at or under 45 deg where you can.",
        "- Estimate the obstacle's world position from the image first (bearing and floor "
        "range formulas above), then choose waypoints, and record both in the Plan so later "
        "turns know where the obstacle was and why each waypoint sits where it does.",
        "- You only see a new frame when you are called. When obstacles are near, arm an "
        "after_s listener (2 to 4 s) or a near_xy listener short of them to look again.",
        "- The frame was taken when the message was sent; the robot keeps moving during your "
        "reply, so allow for the distance covered at the current speed.",
        "- Stairs, curbs and platforms are terrain, not obstacles: declare them with a stairs "
        "or volume Model_Mutation before the walk reaches them, and put the goal on top. The "
        "foot of the first riser is where the bottom step meets the floor: use the floor "
        "formulas above on its lowest visible row. Count the treads.",
        "- Measure the riser height; do not guess it. A riser is a vertical face, so its bottom "
        "edge (where it meets the floor or the tread below) and its top edge (the nosing, where "
        "it meets the tread above) are the same distance ahead. In one column through the "
        "middle of the riser, read the bottom-edge row v_b and the top-edge row v_t (v_t < v_b). "
        "For a row v: yn = (v - cy) / fy and r(v) = (sin(p) + yn cos(p)) / (cos(p) - yn sin(p)), "
        "the tangent of that ray's angle below horizontal. Then",
        "  riser = h (1 - r(v_t) / r(v_b))",
        "  with h and p from CAMERA FRAME. For step k, use its own bottom and top edges and h minus "
        "the height of the tread its bottom edge stands on.",
        "  The floor formulas assume the point is on the floor. For a point on a raised surface of "
        "height t (a tread's far edge, the back edge of a platform), use h - t in place of h: "
        "ahead X = (h - t)(cos(p) - yn sin(p)) / d, and likewise for Y. Measure the riser first, "
        "then place the raised edges with it.",
        "  Scale: with this camera one image row is about 0.6 cm of riser 0.7 to 1.6 m ahead, so a "
        "0.10 m riser spans about 15 to 17 rows. Read the edges to the pixel; a 3-row misread is a "
        "2 cm error. Measure from a frame with the first riser 0.7 to 1.6 m ahead (closer, its "
        "bottom edge leaves the image) and re-measure from the staging point. Write v_b, v_t and "
        "the resulting riser in Analysis.",
        "  The robot measures every tread it steps on and corrects the belief. When the edges are "
        "unclear, round the riser down rather than up: a riser declared too high can make the "
        "robot fall.",
        "- Do not plan to step onto movable objects: route around them.",
        "- The robot has no touch sensing. Pressing on an object also raises |bias|; that is an "
        "external load, not a model error. If |bias| rises while the frame shows the robot "
        "against an object, back off or steer around before queueing any belief search.",
        "- Objects beyond about 6 m are a few pixels wide. Do not infer free space far away "
        "from an empty-looking image.",
    ]
    return "\n".join(lines)


def _ego_mount() -> tuple[float, float, float]:
    from loka.vision import EGO_MOUNT_POS

    return EGO_MOUNT_POS


def format_camera_frame(pose, spec) -> str:
    """The per-turn CAMERA FRAME section that accompanies the attached image."""
    return (
        "## CAMERA FRAME\n"
        f"Attached image: ego_cam, {spec.width} x {spec.height} px RGB, captured at "
        f"t={pose.time:.2f}s when this message was sent.\n"
        f"{pose.describe()}\n"
        f"Camera height above the floor h = {pose.xyz[2]:.3f} m. The robot keeps moving "
        "while you reply."
    )


def build_stand_robot_context(controller: LocomotionController) -> str:
    model = controller.robot.model
    xml_path = Path(controller.config.model_path)
    base = build_robot_model_context(model, str(xml_path))
    limits = controller.lean_limits()
    extra = [
        "",
        "## G1 STAND NOTES",
        f"Nominal CoM height above soles: {controller.nominal_height:.3f} m",
        f"Live lean limits [m]: "
        f"x in [{limits['lean_x_min']:.3f}, {limits['lean_x_max']:.3f}], "
        f"y in [{limits['lean_y_min']:.3f}, {limits['lean_y_max']:.3f}]",
        "qpos[0]=x (pitch), qpos[1]=y (roll), qpos[2]=z (height). "
        "qpos[3:7]=quat wxyz, qpos[7:]=29 joints.",
        "qvel[0:3]=linear, qvel[3:6]=angular, qvel[6:]=joint rates.",
    ]
    # Actuator name list helps amputation.
    names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(f"{i}:{name or '?'}")
    extra.append("Actuators: " + ", ".join(names[:16]) + ", ...")
    return base + "\n" + "\n".join(extra)
