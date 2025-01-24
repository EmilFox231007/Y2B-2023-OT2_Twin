import numpy as np
import cv2
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from simple_pid import PID

def crop_petri_dish(image):
    """ Processes the image to extract and crop the petri dish. """

    # converting to grayscale if the image has multiple channels
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # applying gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # applying thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # extracting contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # removing excessively large contours that may represent image borders
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < image.shape[0] * image.shape[1] * 0.95]

    # identifying the largest contour, which represents the petri dish
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # adjusting bounding box margins for better cropping
    margin = 150
    x += margin
    y += margin
    w -= 2 * margin
    h -= 2 * margin

    # Ensurering cropping coordinates remain within image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Crop the image to focus on the petri dish
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image, (x, y, w, h)

# Initializing the simulation environment with rendering enabled
env = OT2Env(render=True)

# retrieving the image of the petri dish from the simulation
image_path = env.sim.get_plate_image()
plate_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# cropping the image to isolate the petri dish
plate_image, bbox = crop_petri_dish(plate_image)

# Saveing the cropped image
cv2.imwrite("cropped_petri_dish.png", plate_image)

# extract the pixel size of the petri dish for scaling
plate_size_pixels = plate_image.shape[0]  

# fefining the reference position of the petri dish in the robot's coordinate system
plate_position_robot = np.array([0.10775, 0.088 - 0.026, 0.057])  
plate_size_mm = 150  

# computing the conversion factor from pixels to real-world measurements
conversion_factor = plate_size_mm / plate_size_pixels  

# the movement boundaries for the robot
PLATE_MIN_X = plate_position_robot[0]
PLATE_MAX_X = plate_position_robot[0] + (plate_size_mm / 1000)
PLATE_MIN_Y = plate_position_robot[1] - (plate_size_mm / 1000)
PLATE_MAX_Y = plate_position_robot[1]
PLATE_Z = plate_position_robot[2]


def convert_pixel_to_robot_coordinates(pixel_coords):
    """ Converts pixel coordinates from the image to real-world robot coordinates. """
    x_pixels, y_pixels = pixel_coords

    # converming pixel positions to millimeters and then meters
    root_tip_mm_X = x_pixels * conversion_factor / 1000 + plate_position_robot[0]
    root_tip_mm_Y = y_pixels * conversion_factor / 1000 + plate_position_robot[1]

    # Clip values to ensure they remain within the defined plate boundaries
    root_tip_mm_X = np.clip(root_tip_mm_X, PLATE_MIN_X, PLATE_MAX_X)
    root_tip_mm_Y = np.clip(root_tip_mm_Y, PLATE_MIN_Y, PLATE_MAX_Y)

    return np.array([root_tip_mm_X, root_tip_mm_Y, PLATE_Z])


def move_to_goal(env, goal_position, run_index, pipette_position, drop=False, last=False):
    """ Moves the pipette to the target position using PID control. Optionally performs a drop at the goal. """

    print(f"\nRun {run_index + 1}: Moving from {pipette_position} to goal {goal_position}")

    # initializing PID controllers for each axis
    pid_x = PID(100.0, 0.05, 2.0, setpoint=goal_position[0], output_limits=(-1, 1))
    pid_y = PID(100.0, 0.05, 1.5, setpoint=goal_position[1], output_limits=(-1, 1))
    pid_z = PID(50.0, 0.05, 2.0, setpoint=goal_position[2], output_limits=(-1, 1))

    # defining movement termination threshold
    error_threshold = 0.0005  

    # executing movement loop
    for step in range(300):
        current_x, current_y, current_z = pipette_position

        # computing PID output for each axis
        control_x = pid_x(current_x)
        control_y = pid_y(current_y)
        control_z = pid_z(current_z)

        drop_command = 0  

        # applying the action and update state
        action = np.array([control_x, control_y, control_z, drop_command])  
        state, _, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
        pipette_position = state[:3]

        # computing error (distance to goal)
        distance = np.linalg.norm(goal_position - pipette_position)

        print(f"Step {step}: X={current_x:.5f}, Y={current_y:.5f}, Z={current_z:.5f}, Drop: {drop_command}, Error={distance:.6f}")

        # stopping movement when the pipette reaches the goal
        if distance < error_threshold:
            print(f"Goal {goal_position} reached in {step} steps.")

            # performing liquid dispensing if required
            if drop:
                print("Dropping liquid at the root tip.")

                drop_duration = 1 if not last else 1  

                for _ in range(drop_duration):  
                    drop_action = np.array([0, 0, 0, 1])  
                    env.step(drop_action)  

                wait_steps = 10 if not last else 50  

                for _ in range(wait_steps):
                    env.step(np.array([0, 0, 0, 0]))  

            return pipette_position

    print(f"Run {run_index + 1} complete.")
    return pipette_position  


# resetting environment and retrieve initial pipette position
state, _ = env.reset()
pipette_position = state[:3]

# defining target root tip positions in pixel coordinates
coordinates_nodes = [(1318, 372), (1072, 818), (1064, 1322), (1039, 1799), (1090, 2360)]  

# converting pixel coordinates to robot-space coordinates
coordinates_plants = []
for i in coordinates_nodes:
    root_tip_mm_X = i[0] * conversion_factor / 1000 + plate_position_robot[0]
    root_tip_mm_Y = i[1] * conversion_factor / 1000 + plate_position_robot[1]
    coordinates_plants.append([root_tip_mm_X, root_tip_mm_Y, 0.1695])  

# move the pipette to each root position and dispense liquid
for index, robot_goal in enumerate(coordinates_plants):
    is_last = index == len(coordinates_plants) - 1  
    print(f"Moving to Root at Robot Coords: {robot_goal}")
    pipette_position = move_to_goal(env, robot_goal, index, pipette_position, drop=True, last=is_last)  

env.close()
