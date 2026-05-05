"""
Plate Controller — receives rotation commands from the UR5e arm controller.
Commands:
  ROTATE_90   → rotate exactly 90° (π/2 rad) then stop, reply ROTATE_DONE
  PLATE_STOP  → stop immediately
"""
import math
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ── Motor & Sensor ──
motor = robot.getDevice("plate_motor")
sensor = robot.getDevice("plate_sensor")
sensor.enable(timestep)

motor.setPosition(float('inf'))   # velocity mode initially
motor.setVelocity(0)              # start stopped

# ── Comms ──
receiver = robot.getDevice("receiver")
receiver.enable(timestep)
emitter = robot.getDevice("emitter")

# ── State ──
rotating = False
target_position = 0.0
ROTATION_SPEED = 7.0       # rad/s while rotating
POSITION_THRESHOLD = 0.05     # rad — close-enough to target

print("[PLATE] Ready — waiting for commands on channel 1.")

while robot.step(timestep) != -1:
    current = sensor.getValue()

    # ── Check commands ──
    while receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()

        if msg == "ROTATE_90":
            target_position = current + math.pi / 2.0
            motor.setPosition(target_position)
            motor.setVelocity(ROTATION_SPEED)
            rotating = True
            print(f"[PLATE] Rotating 90° → target {target_position:.3f} rad")

        elif msg == "ROTATE_180":
            target_position = current + math.pi
            motor.setPosition(target_position)
            motor.setVelocity(ROTATION_SPEED)
            rotating = True
            print(f"[PLATE] Rotating 180° → target {target_position:.3f} rad")

        elif msg == "PLATE_STOP":
            motor.setVelocity(0)
            rotating = False
            print("[PLATE] Stopped")

    # ── Check if rotation is complete ──
    if rotating:
        if abs(current - target_position) < POSITION_THRESHOLD:
            motor.setVelocity(0)
            rotating = False
            emitter.send("ROTATE_DONE".encode('utf-8'))
            print(f"[PLATE] Rotation complete at {current:.3f} rad")