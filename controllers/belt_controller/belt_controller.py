"""
Belt Controller — listens for STOP / START commands from UR5e via Receiver.
"""
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ── Motor ──
motor = robot.getDevice("belt motor")
motor.setPosition(float('inf'))

# ── Receiver (channel 1, from UR5e emitter) ──
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

BELT_SPEED = 0.9
motor.setVelocity(BELT_SPEED)

print("[BELT] Running — waiting for commands on channel 1.")

while robot.step(timestep) != -1:
    # Check for incoming commands
    while receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()

        if msg == "BELT_STOP":
            motor.setVelocity(0)
            print("[BELT] STOPPED")
        elif msg == "BELT_START":
            motor.setVelocity(BELT_SPEED)
            print("[BELT] STARTED")