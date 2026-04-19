# epuck_spinner.py
# Test controller: spins the e-puck (and plate + box) in place.
# Verifies that the box rotates without falling off.

from controller import Robot
import math

# --- Init ---
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())  # typically 32ms

# E-puck has two motors: left and right wheel
left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

# Set to velocity mode (position = infinity means no target position)
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0)
right_motor.setVelocity(0)

# E-puck position sensors (track wheel rotation)
left_sensor  = robot.getDevice("left wheel sensor")
right_sensor = robot.getDevice("right wheel sensor")
left_sensor.enable(TIME_STEP)
right_sensor.enable(TIME_STEP)

# E-puck specs
WHEEL_RADIUS      = 0.0205   # meters
AXLE_LENGTH       = 0.052    # meters (distance between wheels)
SPIN_SPEED        = 2.0      # rad/s — slow enough box won't fly off
TARGET_ROTATIONS  = 1.0      # how many full 360° spins before reversing

# Arc length one wheel travels for one full robot rotation:
# robot_circumference = pi * axle_length
# wheel_angle_needed  = robot_circumference / wheel_radius
WHEEL_ANGLE_PER_ROBOT_ROT = (math.pi * AXLE_LENGTH) / WHEEL_RADIUS
TARGET_WHEEL_ANGLE = TARGET_ROTATIONS * WHEEL_ANGLE_PER_ROBOT_ROT

print(f"[INIT] Target wheel angle for {TARGET_ROTATIONS} rotation(s): "
      f"{TARGET_WHEEL_ANGLE:.3f} rad ({math.degrees(TARGET_WHEEL_ANGLE):.1f}°)")

# --- State machine ---
STATE_SPIN_CW  = "spin_clockwise"
STATE_STOP     = "stopped"
STATE_SPIN_CCW = "spin_counter"
STATE_DONE     = "done"

state            = STATE_SPIN_CW
start_left       = None
start_right      = None
direction_count  = 0

def set_spin(speed_left, speed_right):
    left_motor.setVelocity(speed_left)
    right_motor.setVelocity(speed_right)

def wheel_delta(current, start):
    """Absolute angle traveled since start."""
    return abs(current - start)

# --- Main loop ---
while robot.step(TIME_STEP) != -1:
    l = left_sensor.getValue()
    r = right_sensor.getValue()

    # Record starting angle when we enter a spin state
    if start_left is None:
        start_left  = l
        start_right = r

    delta_l = wheel_delta(l, start_left)
    delta_r = wheel_delta(r, start_right)
    avg_delta = (delta_l + delta_r) / 2.0

    # How many degrees has the robot body rotated?
    robot_angle_rad = (avg_delta * WHEEL_RADIUS) / (AXLE_LENGTH / 2)
    robot_angle_deg = math.degrees(robot_angle_rad)

    if state == STATE_SPIN_CW:
        set_spin(-SPIN_SPEED, SPIN_SPEED)   # left backward, right forward = CW
        print(f"[CW]  Robot rotated: {robot_angle_deg:6.1f}° / "
              f"{TARGET_ROTATIONS * 360:.0f}°", end="\r")

        if avg_delta >= TARGET_WHEEL_ANGLE:
            set_spin(0, 0)
            state       = STATE_STOP
            start_left  = None
            start_right = None
            print(f"\n[STOP] Completed clockwise rotation. Pausing 1 second...")

    elif state == STATE_STOP:
        # Brief pause — count steps (TIME_STEP ms each)
        if not hasattr(robot, '_pause_steps'):
            robot._pause_steps = 0
        robot._pause_steps += 1
        if robot._pause_steps >= (1000 // TIME_STEP):   # ~1 second
            robot._pause_steps = 0
            direction_count += 1
            if direction_count < 3:   # spin back and forth 3 times total
                state = STATE_SPIN_CCW
                print("[SPIN] Now spinning counter-clockwise...")
            else:
                state = STATE_DONE
                print("[DONE] Test complete. Box stayed on plate!")

    elif state == STATE_SPIN_CCW:
        set_spin(SPIN_SPEED, -SPIN_SPEED)   # reverse direction
        print(f"[CCW] Robot rotated: {robot_angle_deg:6.1f}° / "
              f"{TARGET_ROTATIONS * 360:.0f}°", end="\r")

        if avg_delta >= TARGET_WHEEL_ANGLE:
            set_spin(0, 0)
            state       = STATE_STOP
            start_left  = None
            start_right = None
            print(f"\n[STOP] Completed counter-clockwise rotation. Pausing...")

    elif state == STATE_DONE:
        set_spin(0, 0)   # stay still, test finished