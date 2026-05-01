from controller import Robot, Camera

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get and enable both cameras
top_camera = robot.getDevice("top_camera")
top_camera.enable(timestep)

side_camera = robot.getDevice("side_camera")
side_camera.enable(timestep)

while robot.step(timestep) != -1:
    # Get raw image data
    top_image = top_camera.getImage()
    side_image = side_camera.getImage()
    
    # Get dimensions
    top_w = top_camera.getWidth()
    top_h = top_camera.getHeight()