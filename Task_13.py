import numpy as np
import cv2
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from simple_pid import PID

def crop_petri_dish(image):

    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # Apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # Apply binary threshold
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY) 
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out very large contours which might be the image borders
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < image.shape[0] * image.shape[1] * 0.95]

    # Find the largest contour which will be the petri dish
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Modify the bounding rectangle to crop more tightly
    margin = 150  # Adjust margin size as needed
    x += margin
    y += margin
    w -= 2 * margin
    h -= 2 * margin
    
    # Ensure the modified coordinates are within image bounds
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Crop the image based on calculated coordinates
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image, (x, y, w, h)

# ✅ Initialize OT2 Environment
env = OT2Env(render=True)

# ✅ Retrieve Plate Image
image_path = env.sim.get_plate_image()
plate_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plate_image, bbox = crop_petri_dish(plate_image)
cv2.imwrite("cropped_petri_dish.png", plate_image)


plate_size_pixels = plate_image.shape[0]  # **Use height for scaling**

# ✅ Define Plate Position in Robot Coordinates
plate_position_robot = np.array([0.10775, 0.088 - 0.026, 0.057])  # (X, Y, Z)
plate_size_mm = 150  # Plate size in mm

# ✅ Compute Conversion Factor (mm per pixel)
conversion_factor = plate_size_mm / plate_size_pixels  # mm per pixel

# ✅ Define Plate Boundaries
PLATE_MIN_X = plate_position_robot[0]
PLATE_MAX_X = plate_position_robot[0] + (plate_size_mm / 1000)
PLATE_MIN_Y = plate_position_robot[1] - (plate_size_mm / 1000)
PLATE_MAX_Y = plate_position_robot[1]
PLATE_Z = plate_position_robot[2]

def convert_pixel_to_robot_coordinates(pixel_coords):
    """ Converts pixel coordinates (from the plate image) into robot-space coordinates. """
    x_pixels, y_pixels = pixel_coords

    # ✅ Convert pixel positions to mm and then meters
    root_tip_mm_X = x_pixels * conversion_factor / 1000 + plate_position_robot[0]
    root_tip_mm_Y = y_pixels * conversion_factor / 1000 + plate_position_robot[1]

    # ✅ Debugging: Print conversion results
    print(f"Pixel Coords: {pixel_coords} → Robot Coords: [{root_tip_mm_X:.5f}, {root_tip_mm_Y:.5f}, {PLATE_Z:.5f}]")

    # ✅ Clip within plate boundaries
    root_tip_mm_X = np.clip(root_tip_mm_X, PLATE_MIN_X, PLATE_MAX_X)
    root_tip_mm_Y = np.clip(root_tip_mm_Y, PLATE_MIN_Y, PLATE_MAX_Y)

    return np.array([root_tip_mm_X, root_tip_mm_Y, PLATE_Z])

def move_to_goal(env, goal_position, run_index, pipette_position, drop=False, last=False):
    """ Moves the pipette to a goal using PID control. Drops liquid if drop=True. """

    print(f"\n🔹 Run {run_index + 1}: Moving from {pipette_position} to goal {goal_position}")

    # ✅ PID Controllers
    pid_x = PID(40.0, 0.05, 2.0, setpoint=goal_position[0], output_limits=(-1, 1))
    pid_y = PID(40.0, 0.05, 1.5, setpoint=goal_position[1], output_limits=(-1, 1))
    pid_z = PID(30.0, 0.05, 2.0, setpoint=goal_position[2], output_limits=(-1, 1))

    error_threshold = 0.0005  # Threshold to stop

    # ✅ Move the Pipette Towards the Goal
    for step in range(300):
        current_x, current_y, current_z = pipette_position

        # Compute PID Control Signals
        control_x = pid_x(current_x)
        control_y = pid_y(current_y)
        control_z = pid_z(current_z)

        drop_command = 0  # Default: No drop

        # ✅ No Scaling Applied
        action = np.array([control_x, control_y, control_z, drop_command])  

        # ✅ Apply Action and Update State
        state, _, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
        pipette_position = state[:3]

        # Compute Error (Distance to Goal)
        distance = np.linalg.norm(goal_position - pipette_position)

        print(f"Step {step}: X={current_x:.5f}, Y={current_y:.5f}, Z={current_z:.5f}, Drop: {drop_command}, Error={distance:.6f}")

        # ✅ Stop if Goal is Reached
        if distance < error_threshold:
            print(f"✅ Goal {goal_position} reached in {step} steps!")

            # ✅ Inoculate (Drop) at the Root Tip
            if drop:
                print("💧 Dropping liquid at root tip!")

                drop_duration = 1 if not last else 1  # **Increase drop duration for last drop**
                
                for _ in range(drop_duration):  
                    drop_action = np.array([0, 0, 0, 1])  
                    env.step(drop_action)  

                # ✅ Pause longer for last root
                wait_steps = 10 if not last else 50  # **Increase wait time for last drop**
                
                for _ in range(wait_steps):
                    env.step(np.array([0, 0, 0, 0]))  

            return pipette_position

    print(f"✅ Run {run_index + 1} complete.")
    return pipette_position  

# ✅ Initialize the Environment and Reset State
state, _ = env.reset()
pipette_position = state[:3]

# ✅ **Pipeline for Getting Root Coordinates**
coordinates_nodes = [(1318, 372), (1072, 818), (1064, 1322), (1039, 1799), (1090, 2360)]  

# ✅ Convert Root Pixel Coordinates to Robot 3D Space
coordinates_plants = []
for i in coordinates_nodes:
    root_tip_mm_X = i[0] * conversion_factor / 1000 + plate_position_robot[0]
    root_tip_mm_Y = i[1] * conversion_factor / 1000 + plate_position_robot[1]
    coordinates_plants.append([root_tip_mm_X, root_tip_mm_Y, 0.1695])  

# ✅ Move to Each Root Position and Drop Liquid
for index, robot_goal in enumerate(coordinates_plants):
    is_last = index == len(coordinates_plants) - 1  
    print(f"Moving to Root at Robot Coords: {robot_goal}")
    pipette_position = move_to_goal(env, robot_goal, index, pipette_position, drop=True, last=is_last)  

# ✅ Close the Simulation
env.close()
