from controller import Robot

# Init robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get motor
motor = robot.getDevice("belt motor")

# Set velocity mode
motor.setPosition(float('inf'))

# Set speed
speed = 0.1
motor.setVelocity(speed)

# Main loop
while robot.step(timestep) != -1:
    pass