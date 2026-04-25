from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

# Set to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Spin in place: one wheel forward, one backward
left_motor.setVelocity(6.28)   # max speed forward
right_motor.setVelocity(-6.28) # max speed backward

while robot.step(timestep) != -1:
    pass