from controller import Robot

robot    = Robot()
timestep = int(robot.getBasicTimeStep())

motor = robot.getDevice("plate_motor")
motor.setPosition(float('inf'))  # velocity mode
motor.setVelocity(1.5)           # radians/second — change this to go faster/slower

while robot.step(timestep) != -1:
    pass  # plate spins forever