"""
ARM CONTROLLER — Full FSM for QR Code Inspection & Sorting
==========================================================
States:
  IDLE              → monitor top camera for box on conveyor
  BOX_DETECTED      → stop belt, prepare pickup
  MOVE_TO_PICKUP    → move arm above conveyor box
  GRAB_BOX          → lower & close gripper
  LIFT_BOX          → lift box from conveyor
  MOVE_TO_PLATE     → move arm to above plate
  PLACE_ON_PLATE    → lower & release box on plate
  RETRACT           → pull arm away so camera has clear view
  SCAN_FACE         → use side camera + OpenCV QR detector
  REQUEST_ROTATE    → ask plate to rotate 90°
  WAIT_ROTATE       → wait for ROTATE_DONE
  FLIP_GRAB         → grab box from plate to flip it
  FLIP_LIFT         → lift box for flipping
  FLIP_ROTATE       → rotate wrist to flip box 180°
  FLIP_PLACE        → place flipped box back on plate
  SCAN_BOTTOM       → scan bottom face (now facing side cam)
  ROTATE_FOR_TOP    → rotate plate 90° to see old top face
  SCAN_TOP          → scan top face
  PICK_FOR_SORT     → grab box from plate for final sorting
  LIFT_FOR_SORT     → lift from plate
  MOVE_TO_GOOD      → move to RectangleArena (QR found)
  PLACE_GOOD        → place in blue area
  MOVE_TO_REJECT    → move to MetalStorageBox (no QR)
  PLACE_REJECT      → drop in metal box
  RETURN_HOME       → go back to home, resume belt
"""
import sys
import math
import numpy as np
import cv2

# ── Webots imports ──
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ══════════════════════════════════════════════
# DEVICES
# ══════════════════════════════════════════════

# ── Cameras ──
top_camera = robot.getDevice("top_camera")
top_camera.enable(timestep)

side_camera = robot.getDevice("side_camera")
side_camera.enable(timestep)

# ── Emitter / Receiver ──
emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

# ── Arm joints ──
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
motors = []
sensors = []
for name in JOINT_NAMES:
    m = robot.getDevice(name)
    s = robot.getDevice(name + "_sensor")
    s.enable(timestep)
    m.setVelocity(1.0)          # default speed (rad/s)
    motors.append(m)
    sensors.append(s)

# ── Gripper ──
finger_left = robot.getDevice("finger motor::left")
finger_right = robot.getDevice("finger motor::right")
lift_motor = robot.getDevice("lift motor")

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.005

# ── OpenCV QR Detector ──
qr_detector = cv2.QRCodeDetector()

# ══════════════════════════════════════════════
# JOINT POSITIONS  (shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3)
# ══════════════════════════════════════════════
# NOTE: These are initial estimates. You WILL need to tune them
# in Webots by moving the robot manually and reading sensor values.

POS_HOME        = [0.0,    -1.5708, 0.0,    -1.5708, 0.0,    0.0]

# Conveyor pickup — BASE template (shoulder_pan + lift will be overridden dynamically)
POS_ABOVE_CONV  = [1.20,   -1.10,   0.90,   -1.40,  -1.5708, 0.0]
POS_PICK_CONV   = [1.20,   -0.85,   0.90,   -1.65,  -1.5708, 0.0]

# Plate — arm above rotating plate
POS_ABOVE_PLATE = [0.0,    -1.10,   0.90,   -1.40,  -1.5708, 0.0]
POS_ON_PLATE    = [0.0,    -0.75,   0.80,   -1.65,  -1.5708, 0.0]

# Retracted — arm out of camera view while scanning
POS_RETRACT     = [0.0,    -1.5708, 0.0,    -1.5708, 0.0,    0.0]

# Flip positions — grab, lift, rotate wrist 180°, place back
POS_FLIP_GRAB   = POS_ON_PLATE[:]
POS_FLIP_LIFT   = [0.0,    -1.10,   0.90,   -1.40,  -1.5708, 0.0]
POS_FLIP_ROTATED = [0.0,   -1.10,   0.90,   -1.40,  -1.5708, 3.1416]  # wrist3 rotated 180°
POS_FLIP_PLACE  = [0.0,    -0.75,   0.80,   -1.65,  -1.5708, 3.1416]

# Good box destination — RectangleArena (blue area at world -0.43, 0.83)
POS_ABOVE_GOOD  = [-1.30,  -1.10,   0.80,   -1.30,  -1.5708, 0.0]
POS_PLACE_GOOD  = [-1.30,  -0.85,   0.80,   -1.55,  -1.5708, 0.0]

# Reject destination — MetalStorageBox (world 0.43, 1.74)
POS_ABOVE_REJECT = [3.00,  -1.30,   0.60,   -0.90,  -1.5708, 0.0]
POS_PLACE_REJECT = [3.00,  -1.05,   0.60,   -1.15,  -1.5708, 0.0]

# ══════════════════════════════════════════════
# TOP CAMERA → WORLD COORDINATE MAPPING
# ══════════════════════════════════════════════
# Camera world position (computed from UR5e base + camera local offset + rotation)
CAM_WORLD_X = 1.079
CAM_WORLD_Y = 0.916
CAM_WORLD_Z = 0.689

# Conveyor surface Z where box center sits (belt at 0.156 + half box 0.025)
CONV_BOX_Z = 0.18

# Camera geometry
CAM_HEIGHT_ABOVE_BOX = CAM_WORLD_Z - CONV_BOX_Z       # ≈ 0.51m
CAM_FOV    = 1.0       # radians
CAM_W      = 1280      # pixels
CAM_H      = 720       # pixels
PIX_PER_M  = CAM_W / (2.0 * CAM_HEIGHT_ABOVE_BOX * math.tan(CAM_FOV / 2.0))

# UR5e base position in world
UR5E_X = 0.39
UR5E_Y = 1.02


def pixel_to_world(cx, cy):
    """Convert top-camera pixel coordinates to approximate world (X, Y)
    on the conveyor plane."""
    dx_px = cx - CAM_W / 2.0
    dy_px = cy - CAM_H / 2.0
    dx_m = dx_px / PIX_PER_M
    dy_m = dy_px / PIX_PER_M
    # Image axes → world axes (accounting for camera + UR5e rotations)
    # Image X ≈ world +Y direction, Image Y ≈ world +X direction
    # (signs may need tuning based on actual camera orientation)
    world_x = CAM_WORLD_X + dy_m
    world_y = CAM_WORLD_Y - dx_m
    return world_x, world_y


def compute_pickup_joints(world_x, world_y, template):
    """Compute arm joint angles to reach a world (X, Y) point on conveyor.
    Uses 'template' as the base joint angles and adjusts shoulder_pan + lift."""
    # Convert world to UR5e local frame
    dx = world_x - UR5E_X
    dy = world_y - UR5E_Y
    # Apply UR5e rotation (+90° to undo the -90° world rotation)
    local_x = -dy
    local_y = dx

    # shoulder_pan = angle to target in local XY
    pan = math.atan2(local_y, local_x)

    # Horizontal reach
    reach = math.sqrt(local_x**2 + local_y**2)
    base_reach = 0.72  # approximate reach at default conveyor position
    reach_diff = reach - base_reach

    angles = template[:]
    angles[0] = pan
    # Adjust shoulder_lift slightly for reach difference
    angles[1] = template[1] + reach_diff * 0.4

    print(f"[IK] world=({world_x:.3f}, {world_y:.3f}) → pan={math.degrees(pan):.1f}° reach={reach:.3f}m")
    return angles

# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def webots_camera_to_cv(camera):
    """Convert Webots camera image (BGRA) to OpenCV RGB numpy array."""
    raw = camera.getImage()
    if raw is None:
        return None, 0, 0
    w = camera.getWidth()
    h = camera.getHeight()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    return rgb, w, h


def move_arm(target, speed=1.0):
    """Set all joint targets. Non-blocking — returns immediately."""
    for i, m in enumerate(motors):
        m.setVelocity(speed)
        m.setPosition(target[i])


def arm_at_target(target, threshold=0.05):
    """Check if all joints have reached their target positions."""
    for i, s in enumerate(sensors):
        if abs(s.getValue() - target[i]) > threshold:
            return False
    return True


def open_gripper():
    finger_left.setPosition(GRIPPER_OPEN)
    finger_right.setPosition(GRIPPER_OPEN)


def close_gripper():
    finger_left.setPosition(GRIPPER_CLOSED)
    finger_right.setPosition(GRIPPER_CLOSED)


def send_command(cmd):
    """Send a string command via Emitter to plate/belt controllers."""
    emitter.send(cmd.encode('utf-8'))
    print(f"[ARM] Sent: {cmd}")


def check_receiver():
    """Check for messages from plate controller."""
    while receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()
        return msg
    return None


def detect_box_top_camera(img_rgb):
    """Detect orange boxes on conveyor via color segmentation and check for QR.
    Returns (detected, qr_found, qr_data, annotated_bgr, centroid_xy or None)."""
    if img_rgb is None:
        return False, False, None, None, None

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated = img_bgr.copy()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Orange range in HSV
    lower = np.array([5, 80, 80])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    best_centroid = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            detected = True
            # Draw contour
            cv2.drawContours(annotated, [c], -1, (0, 255, 0), 2)
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # Centroid
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                # Keep the largest box centroid
                if area > best_area:
                    best_area = area
                    best_centroid = (cx, cy)
            # Label
            cv2.putText(annotated, "Box", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Status overlay
    status = "BOX DETECTED" if detected else "No box"
    color = (0, 255, 0) if detected else (128, 128, 128)
    cv2.putText(annotated, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if best_centroid:
        cv2.putText(annotated, f"pos: ({best_centroid[0]}, {best_centroid[1]})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # QR Detection on top camera
    qr_data, bbox, _ = qr_detector.detectAndDecode(img_bgr)
    qr_found = False
    if bbox is not None and len(bbox) > 0 and qr_data:
        qr_found = True
        pts = bbox[0].astype(int)
        n = len(pts)
        for i in range(n):
            cv2.line(annotated, tuple(pts[i]), tuple(pts[(i+1) % n]), (255, 0, 0), 3) # Blue contour for QR
        cv2.putText(annotated, f"QR DETECTED: {qr_data}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return detected, qr_found, qr_data, annotated, best_centroid


def detect_qr_side_camera():
    """Capture from side camera and detect/decode QR codes.
    Returns (found, data, annotated_frame)."""
    img_rgb, w, h = webots_camera_to_cv(side_camera)
    if img_rgb is None:
        return False, None, None

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated = img_bgr.copy()

    data, bbox, _ = qr_detector.detectAndDecode(img_bgr)

    if bbox is not None and len(bbox) > 0 and data:
        pts = bbox[0].astype(int)
        n = len(pts)
        # Draw filled semi-transparent polygon
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
        # Draw thick polygon outline
        for i in range(n):
            cv2.line(annotated, tuple(pts[i]), tuple(pts[(i+1) % n]),
                     (0, 255, 0), 3)
        # Draw corner circles
        for pt in pts:
            cv2.circle(annotated, tuple(pt), 6, (255, 0, 255), -1)
        # Label with QR data
        cv2.putText(annotated, f"QR: {data}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Bounding rect around QR
        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.rectangle(annotated, (x - 5, y - 5), (x + bw + 5, y + bh + 5),
                      (0, 255, 0), 1)
        found = True
    else:
        cv2.putText(annotated, "No QR detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        found = False
        data = None

    # Always show state info
    cv2.putText(annotated, f"State: {state}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return found, data, annotated


# ══════════════════════════════════════════════
# FINITE STATE MACHINE
# ══════════════════════════════════════════════

state = "IDLE"
prev_state = ""
side_faces_scanned = 0
qr_found = False
qr_data = None
wait_counter = 0          # generic wait/dwell timer
DWELL_TICKS = 20          # how many timesteps to dwell for scanning
GRIP_TICKS = 15           # timesteps to wait for gripper to close/open

# ══════════════════════════════════════════════
# OPENCV WINDOW SETUP — resizable windows
# ══════════════════════════════════════════════
WIN_TOP = "Top Camera - Box Detection"
WIN_SIDE = "Side Camera - QR Detection"
TOP_WIN_SIZE = (640, 360)      # (width, height) — change these to resize
SIDE_WIN_SIZE = (640, 360)     # (width, height) — change these to resize

cv2.namedWindow(WIN_TOP, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_TOP, *TOP_WIN_SIZE)

cv2.namedWindow(WIN_SIDE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_SIDE, *SIDE_WIN_SIZE)

print("=" * 60)
print("[ARM] QR Inspection & Sorting System — STARTING")
print("=" * 60)

open_gripper()
move_arm(POS_HOME)

while robot.step(timestep) != -1:

    # ── State transition logging ──
    if state != prev_state:
        print(f"[FSM] {prev_state} → {state}")
        prev_state = state

    # ── Always show side camera feed with QR contouring ──
    _, _, side_frame = detect_qr_side_camera()
    if side_frame is not None:
        cv2.imshow(WIN_SIDE, side_frame)

    # ── Show top camera feed with box contouring ──
    top_img, tw, th = webots_camera_to_cv(top_camera)
    box_detected_now, top_qr_detected_now, top_qr_data, top_annotated, box_centroid = detect_box_top_camera(top_img)
    if top_annotated is not None:
        cv2.imshow(WIN_TOP, top_annotated)

    cv2.waitKey(1)

    # ══════════════════════════════════════════
    # STATE HANDLERS
    # ══════════════════════════════════════════

    # ────────────────────────────────────────────
    # IDLE — wait for box on conveyor (top camera)
    # ────────────────────────────────────────────
    if state == "IDLE":
        if arm_at_target(POS_HOME):
            if box_detected_now:
                print("[ARM] Box detected on conveyor!")
                wait_counter = 0
                state = "BELT_DELAY"

    # ────────────────────────────────────────────
    # BELT_DELAY — let box travel a tiny bit more before stopping
    # ────────────────────────────────────────────
    elif state == "BELT_DELAY":
        wait_counter += 1
        BELT_DELAY_TICKS = 10    # ~160ms at 16ms timestep — adjust as needed
        if wait_counter >= BELT_DELAY_TICKS:
            send_command("BELT_STOP")
            print("[ARM] Belt stopped after short delay")
            state = "BOX_DETECTED"

    # ────────────────────────────────────────────
    # BOX_DETECTED — init variables, wait for box to settle
    # ────────────────────────────────────────────
    elif state == "BOX_DETECTED":
        qr_found = False
        qr_data = None
        side_faces_scanned = 0
        wait_counter = 0
        state = "BOX_SETTLING"

    # ────────────────────────────────────────────
    # BOX_SETTLING — wait for box to stop moving, then locate it
    # ────────────────────────────────────────────
    elif state == "BOX_SETTLING":
        wait_counter += 1
        
        if top_qr_detected_now:
            qr_found = True
            qr_data = top_qr_data

        SETTLE_TICKS = 50    # ~0.8s at 16ms timestep — let box come to rest
        if wait_counter >= SETTLE_TICKS:
            # Now get the box position from camera
            if box_centroid is not None:
                cx, cy = box_centroid
                bx, by = pixel_to_world(cx, cy)
                # Compute dynamic pickup positions from camera-measured location
                dynamic_above = compute_pickup_joints(bx, by, POS_ABOVE_CONV)
                dynamic_pick  = compute_pickup_joints(bx, by, POS_PICK_CONV)
                print(f"[ARM] Box located at world ({bx:.3f}, {by:.3f}), moving to pick")
                if qr_found:
                    print(f"[QR] ✓ FOUND on TOP camera: {qr_data}")
            else:
                # Fallback to default if centroid lost
                dynamic_above = POS_ABOVE_CONV[:]
                dynamic_pick  = POS_PICK_CONV[:]
                print("[ARM] Centroid lost — using default pickup position")
            state = "MOVE_TO_PICKUP"

    # ────────────────────────────────────────────
    # MOVE_TO_PICKUP — arm above conveyor (dynamic position)
    # ────────────────────────────────────────────
    elif state == "MOVE_TO_PICKUP":
        open_gripper()
        move_arm(dynamic_above)
        state = "WAIT_ABOVE_CONV"

    elif state == "WAIT_ABOVE_CONV":
        if arm_at_target(dynamic_above):
            move_arm(dynamic_pick, speed=0.5)
            state = "LOWERING_TO_CONV"

    elif state == "LOWERING_TO_CONV":
        if arm_at_target(dynamic_pick):
            close_gripper()
            wait_counter = 0
            state = "GRAB_BOX"

    # ────────────────────────────────────────────
    # GRAB_BOX — close gripper, wait, then lift
    # ────────────────────────────────────────────
    elif state == "GRAB_BOX":
        wait_counter += 1
        if wait_counter >= GRIP_TICKS:
            move_arm(dynamic_above)
            state = "LIFT_BOX"

    elif state == "LIFT_BOX":
        if arm_at_target(dynamic_above):
            if qr_found:
                print("[ARM] QR already found from top camera! Direct to GOOD area.")
                state = "MOVE_TO_GOOD"
            else:
                state = "MOVE_TO_PLATE"

    # ────────────────────────────────────────────
    # MOVE_TO_PLATE — carry box to plate
    # ────────────────────────────────────────────
    elif state == "MOVE_TO_PLATE":
        move_arm(POS_ABOVE_PLATE)
        state = "WAIT_ABOVE_PLATE"

    elif state == "WAIT_ABOVE_PLATE":
        if arm_at_target(POS_ABOVE_PLATE):
            move_arm(POS_ON_PLATE, speed=0.5)
            state = "LOWERING_TO_PLATE"

    elif state == "LOWERING_TO_PLATE":
        if arm_at_target(POS_ON_PLATE):
            open_gripper()
            wait_counter = 0
            state = "RELEASE_ON_PLATE"

    elif state == "RELEASE_ON_PLATE":
        wait_counter += 1
        if wait_counter >= GRIP_TICKS:
            state = "RETRACT"

    # ────────────────────────────────────────────
    # RETRACT — move arm out of camera view
    # ────────────────────────────────────────────
    elif state == "RETRACT":
        move_arm(POS_RETRACT)
        state = "WAIT_RETRACT"

    elif state == "WAIT_RETRACT":
        if arm_at_target(POS_RETRACT):
            wait_counter = 0
            state = "SCAN_FACE"

    # ────────────────────────────────────────────
    # SCAN_FACE — check side camera for QR code
    # ────────────────────────────────────────────
    elif state == "SCAN_FACE":
        wait_counter += 1
        if wait_counter >= DWELL_TICKS:
            found, data, _ = detect_qr_side_camera()
            if found:
                qr_found = True
                qr_data = data
                print(f"[QR] ✓ FOUND on side face #{side_faces_scanned + 1}: {data}")
                state = "PICK_FOR_SORT"
            else:
                side_faces_scanned += 1
                print(f"[QR] ✗ Not found — side face {side_faces_scanned}/4")
                if side_faces_scanned < 4:
                    state = "REQUEST_ROTATE"
                else:
                    # 4 sides done, try flipping
                    state = "FLIP_GRAB"

    # ────────────────────────────────────────────
    # ROTATE PLATE 90°
    # ────────────────────────────────────────────
    elif state == "REQUEST_ROTATE":
        send_command("ROTATE_90")
        state = "WAIT_ROTATE"

    elif state == "WAIT_ROTATE":
        msg = check_receiver()
        if msg == "ROTATE_DONE":
            print("[ARM] Plate rotation complete")
            wait_counter = 0
            state = "SCAN_FACE"

    # ────────────────────────────────────────────
    # FLIP BOX — pick from plate, rotate 180°, place back
    # ────────────────────────────────────────────
    elif state == "FLIP_GRAB":
        print("[ARM] Flipping box to check bottom/top faces...")
        move_arm(POS_ABOVE_PLATE)
        state = "FLIP_WAIT_ABOVE"

    elif state == "FLIP_WAIT_ABOVE":
        if arm_at_target(POS_ABOVE_PLATE):
            move_arm(POS_FLIP_GRAB, speed=0.5)
            state = "FLIP_LOWERING"

    elif state == "FLIP_LOWERING":
        if arm_at_target(POS_FLIP_GRAB):
            close_gripper()
            wait_counter = 0
            state = "FLIP_GRIPPING"

    elif state == "FLIP_GRIPPING":
        wait_counter += 1
        if wait_counter >= GRIP_TICKS:
            move_arm(POS_FLIP_LIFT)
            state = "FLIP_LIFTING"

    elif state == "FLIP_LIFTING":
        if arm_at_target(POS_FLIP_LIFT):
            move_arm(POS_FLIP_ROTATED, speed=0.8)
            state = "FLIP_ROTATING"

    elif state == "FLIP_ROTATING":
        if arm_at_target(POS_FLIP_ROTATED):
            move_arm(POS_FLIP_PLACE, speed=0.5)
            state = "FLIP_PLACING"

    elif state == "FLIP_PLACING":
        if arm_at_target(POS_FLIP_PLACE):
            open_gripper()
            wait_counter = 0
            state = "FLIP_RELEASING"

    elif state == "FLIP_RELEASING":
        wait_counter += 1
        if wait_counter >= GRIP_TICKS:
            # Reset wrist3 back to 0 during retract
            retract_after_flip = POS_RETRACT[:]
            move_arm(retract_after_flip)
            state = "FLIP_RETRACT"

    elif state == "FLIP_RETRACT":
        if arm_at_target(POS_RETRACT):
            wait_counter = 0
            state = "SCAN_BOTTOM"

    # ────────────────────────────────────────────
    # SCAN BOTTOM face (box was flipped)
    # ────────────────────────────────────────────
    elif state == "SCAN_BOTTOM":
        wait_counter += 1
        if wait_counter >= DWELL_TICKS:
            found, data, _ = detect_qr_side_camera()
            if found:
                qr_found = True
                qr_data = data
                print(f"[QR] ✓ FOUND on bottom face: {data}")
                state = "PICK_FOR_SORT"
            else:
                print("[QR] ✗ Not found on bottom face — rotating 90° to see top")
                state = "ROTATE_FOR_TOP"

    # ────────────────────────────────────────────
    # ROTATE 90° to see the original top face
    # ────────────────────────────────────────────
    elif state == "ROTATE_FOR_TOP":
        send_command("ROTATE_90")
        state = "WAIT_ROTATE_TOP"

    elif state == "WAIT_ROTATE_TOP":
        msg = check_receiver()
        if msg == "ROTATE_DONE":
            wait_counter = 0
            state = "SCAN_TOP"

    elif state == "SCAN_TOP":
        wait_counter += 1
        if wait_counter >= DWELL_TICKS:
            found, data, _ = detect_qr_side_camera()
            if found:
                qr_found = True
                qr_data = data
                print(f"[QR] ✓ FOUND on top face: {data}")
                state = "PICK_FOR_SORT"
            else:
                print("[QR] ✗ QR NOT FOUND on ANY face → REJECT")
                state = "PICK_FOR_SORT"   # will go to reject based on qr_found flag

    # ────────────────────────────────────────────
    # PICK FROM PLATE for final sorting
    # ────────────────────────────────────────────
    elif state == "PICK_FOR_SORT":
        move_arm(POS_ABOVE_PLATE)
        state = "SORT_WAIT_ABOVE"

    elif state == "SORT_WAIT_ABOVE":
        if arm_at_target(POS_ABOVE_PLATE):
            move_arm(POS_ON_PLATE, speed=0.5)
            state = "SORT_LOWERING"

    elif state == "SORT_LOWERING":
        if arm_at_target(POS_ON_PLATE):
            close_gripper()
            wait_counter = 0
            state = "SORT_GRIPPING"

    elif state == "SORT_GRIPPING":
        wait_counter += 1
        if wait_counter >= GRIP_TICKS:
            move_arm(POS_ABOVE_PLATE)
            state = "SORT_LIFTING"

    elif state == "SORT_LIFTING":
        if arm_at_target(POS_ABOVE_PLATE):
            if qr_found:
                print(f"[ARM] QR found ({qr_data}) → placing in GOOD area")
                state = "MOVE_TO_GOOD"
            else:
                print("[ARM] No QR found → placing in REJECT bin")
                state = "MOVE_TO_REJECT"

    # ────────────────────────────────────────────
    # GOOD — place in RectangleArena (QR found)
    # ────────────────────────────────────────────
    elif state == "MOVE_TO_GOOD":
        move_arm(POS_ABOVE_GOOD)
        state = "WAIT_ABOVE_GOOD"

    elif state == "WAIT_ABOVE_GOOD":
        if arm_at_target(POS_ABOVE_GOOD):
            move_arm(POS_PLACE_GOOD, speed=0.5)
            state = "LOWERING_GOOD"

    elif state == "LOWERING_GOOD":
        if arm_at_target(POS_PLACE_GOOD):
            open_gripper()
            wait_counter = 0
            state = "RELEASE_GOOD"

    elif state == "RELEASE_GOOD":
        wait_counter += 1
        if wait_counter >= GRIP_TICKS:
            move_arm(POS_ABOVE_GOOD)
            state = "LIFT_FROM_GOOD"

    elif state == "LIFT_FROM_GOOD":
        if arm_at_target(POS_ABOVE_GOOD):
            state = "RETURN_HOME"

    # ────────────────────────────────────────────
    # REJECT — place in MetalStorageBox
    # ────────────────────────────────────────────
    elif state == "MOVE_TO_REJECT":
        move_arm(POS_ABOVE_REJECT)
        state = "WAIT_ABOVE_REJECT"

    elif state == "WAIT_ABOVE_REJECT":
        if arm_at_target(POS_ABOVE_REJECT):
            move_arm(POS_PLACE_REJECT, speed=0.5)
            state = "LOWERING_REJECT"

    elif state == "LOWERING_REJECT":
        if arm_at_target(POS_PLACE_REJECT):
            open_gripper()
            wait_counter = 0
            state = "RELEASE_REJECT"

    elif state == "RELEASE_REJECT":
        wait_counter += 1
        if wait_counter >= GRIP_TICKS:
            move_arm(POS_ABOVE_REJECT)
            state = "LIFT_FROM_REJECT"

    elif state == "LIFT_FROM_REJECT":
        if arm_at_target(POS_ABOVE_REJECT):
            state = "RETURN_HOME"

    # ────────────────────────────────────────────
    # RETURN HOME — resume belt, wait for next box
    # ────────────────────────────────────────────
    elif state == "RETURN_HOME":
        move_arm(POS_HOME)
        send_command("BELT_START")
        state = "WAIT_HOME"

    elif state == "WAIT_HOME":
        if arm_at_target(POS_HOME):
            print("[ARM] ── Cycle complete. Waiting for next box. ──")
            state = "IDLE"

cv2.destroyAllWindows()