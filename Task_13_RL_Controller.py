import numpy as np
import cv2
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from stable_baselines3 import PPO  

# ✅ Initialize OT2 Environment
env = OT2Env(render=True)

# ✅ Load Pre-Trained RL Model
model_path = "Astra_V3.0.zip"  # Update this with the correct path
model = PPO.load(model_path)  

# ✅ Retrieve Plate Image
image_path = env.sim.get_plate_image()
plate_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plate_size_pixels = plate_image.shape[0]

# ✅ Define Plate Position in Robot Coordinates
plate_position_robot = np.array([0.10775, 0.088 - 0.026, 0.057])  # (X, Y, Z)
plate_size_mm = 150  # Plate size in mm
conversion_factor = plate_size_mm / plate_size_pixels  # mm per pixel

# ✅ Convert Root Pixel Coordinates to Robot 3D Space
def convert_pixel_to_robot_coordinates(pixel_coords):
    x_pixels, y_pixels = pixel_coords
    root_tip_mm_X = x_pixels * conversion_factor / 1000 + plate_position_robot[0]
    root_tip_mm_Y = y_pixels * conversion_factor / 1000 + plate_position_robot[1]
    return np.array([root_tip_mm_X, root_tip_mm_Y, plate_position_robot[2]])

# ✅ Move to Goal Using RL Model
def move_to_goal(env, goal_position, run_index, observation, drop=False, last=False):
    print(f"\n🔹 Run {run_index + 1}: Moving to goal {goal_position}")

    for step in range(300):  # Limit steps to avoid infinite loops
        print(f"Step {step}: Current Position: {observation[:3]} | Goal: {goal_position}")

        action, _ = model.predict(observation)
        action = np.clip(action, -0.5, 0.5)  # Prevent extreme movements
        print(f"Predicted Action: {action}")

        observation, _, _, _, _ = env.step(action)
        pipette_position = observation[:3]  # Extract pipette position

        distance = np.linalg.norm(goal_position - pipette_position)
        print(f"Error={distance:.6f}")

        if distance < 0.0005:
            print(f"✅ Goal {goal_position} reached in {step} steps!")
            if drop:
                for _ in range(1 if not last else 2):
                    env.step(np.array([0, 0, 0, 1]))  # Execute Drop Action
                for _ in range(10 if not last else 50):
                    env.step(np.array([0, 0, 0, 0]))  # Wait for drop
            return observation  

    print(f"✅ Run {run_index + 1} complete.")
    return observation

# ✅ Initialize the Environment and Reset State
observation, _ = env.reset()  # Full observation (6,)

# ✅ Example Root Positions (Change this with actual detected root tips)
coordinates_nodes = [(1318, 372), (1072, 818), (1064, 1322), (1039, 1799), (1090, 2360)]  
coordinates_plants = [convert_pixel_to_robot_coordinates(i) for i in coordinates_nodes]

# ✅ Move to Each Root Position and Drop Liquid
for index, robot_goal in enumerate(coordinates_plants):
    is_last = index == len(coordinates_plants) - 1  
    print(f"Moving to Root at Robot Coords: {robot_goal}")
    observation = move_to_goal(env, robot_goal, index, observation, drop=True, last=is_last)  

# ✅ Close the Simulation
env.close()