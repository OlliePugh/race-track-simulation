from controller import Robot, Motor, Keyboard

# Create a robot instance and initialize it
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize the motors
left_motor = robot.getDevice('left_wheel_motor')
right_motor = robot.getDevice('right_wheel_motor')
left_servo = robot.getDevice('front_left_servo')
right_servo = robot.getDevice('front_right_servo')

# Set the target position of the motors
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_servo.setPosition(float('inf'))
right_servo.setPosition(float('inf'))

# Set the velocity of the motors
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera = robot.getDevice('camera')
camera.enable(timestep)

# Set the camera properties
camera_width = camera.getWidth()
camera_height = camera.getHeight()
camera_fov = camera.getFov()
camera_framerate = 24

# Initialize the keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

#Main control loop
while robot.step(timestep) != -1:
#    Read the keyboard input
    keys_down = []
    keys_down.append(keyboard.getKey())
    keys_down.append(keyboard.getKey())
    forward_speed = 60

 #   Move the car based on the keyboard input
    if Keyboard.UP in keys_down:
        left_motor.setVelocity(forward_speed)
        right_motor.setVelocity(forward_speed)
    elif Keyboard.DOWN in keys_down:
        left_motor.setVelocity(-10.0)
        right_motor.setVelocity(-10.0)
    else:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        
    if Keyboard.LEFT in keys_down:
        left_servo.setPosition(0.5)
        right_servo.setPosition(0.5)
    elif Keyboard.RIGHT in keys_down:
        left_servo.setPosition(-0.5)
        right_servo.setPosition(-0.5)
    else:
        left_servo.setPosition(0)
        right_servo.setPosition(0)
    
    