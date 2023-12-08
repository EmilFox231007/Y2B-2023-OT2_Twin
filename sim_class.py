import pybullet as p
import time
import pybullet_data
import math
import logging

logging.basicConfig(level=logging.INFO)

class Simulation:
    def __init__(self, num_agents):
        # Set up the simulation
        self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        # Hide the default GUI components
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        #p.setPhysicsEngineParameter(contactBreakingThreshold=0.000001)

        # Set the camera parameters
        cameraDistance = 1.1*(math.ceil((num_agents)**0.3)) # Distance from the target (zoom)
        cameraYaw = 90  # Rotation around the vertical axis in degrees
        cameraPitch = -35  # Rotation around the horizontal axis in degrees
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents**0.5)/2)+0.5, 0.1]  # XYZ coordinates of the target position

        # Reset the camera with the specified parameters
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        self.baseplaneId = p.loadURDF("plane.urdf")
        # add collision shape to the plane
        #p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[30, 305, 0.001])

        # Create the robots
        self.create_robots(num_agents)

        # define the pipette offset
        self.pipette_offset = [0.073, 0.0895, 0.0895]
    
    # method to create n robots in a grid pattern
    def create_robots(self, num_agents):
        spacing = 1  # Adjust the spacing as needed

        # Calculate the grid size to fit all agents
        grid_size = math.ceil(num_agents ** 0.5) 

        self.robotIds = []
        self.specimenIds = []
        agent_count = 0  # Counter for the number of placed agents

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count < num_agents:  # Check if more agents need to be placed
                    # Calculate position for each robot
                    position = [-spacing * i, -spacing * j, 0.03]
                    robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0,0,0,1],
                                        flags=p.URDF_USE_INERTIA_FROM_FILE)
                    start_position, start_orientation = p.getBasePositionAndOrientation(robotId)
                    p.createConstraint(parentBodyUniqueId=robotId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=start_position,
                                    childFrameOrientation=start_orientation)

                    # Create a fixed constraint between the robot and the base plane so the robot is fixed in space above the plane by its base link with an offset
                    #p.createConstraint(self.baseplaneId, -1, robotId, -1, p.JOINT_FIXED, [0, 0, 0], position, [0, 0, 0])

                    # Load the specimen with an offset
                    offset = [0.18275, 0.163, 0.057]
                    position_with_offset = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                    rotate_90 = p.getQuaternionFromEuler([0, 0, -math.pi/2])
                    planeId = p.loadURDF("custom.urdf", position_with_offset, rotate_90)#start_orientation)
                    # Disable collision between the robot and the specimen
                    p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                    spec_position, spec_orientation = p.getBasePositionAndOrientation(planeId)

                    #Constrain the specimen to the robot
                    # p.createConstraint(parentBodyUniqueId=robotId,
                    #                 parentLinkIndex=-1,
                    #                 childBodyUniqueId=planeId,
                    #                 childLinkIndex=-1,
                    #                 jointType=p.JOINT_FIXED,
                    #                 jointAxis=[0, 0, 0],
                    #                 parentFramePosition=start_position,
                    #                 #parentFrameOrientation=start_orientation,
                    #                 childFramePosition=[0, 0, 0],
                    #                 childFrameOrientation=[0, 0, 0, 1])
                    #p.createConstraint(robotId, -1, planeId, -1, p.JOINT_FIXED, [0, 0, 0], offset, [0, 0, 0])
                    p.createConstraint(parentBodyUniqueId=planeId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=spec_position,
                                    childFrameOrientation=spec_orientation)
                    # Load your texture and apply it to the plane
                    textureId = p.loadTexture("uvmapped_dish_large_comp.png")
                    p.changeVisualShape(planeId, -1, textureUniqueId=textureId)

                    self.robotIds.append(robotId)
                    self.specimenIds.append(planeId)

                    agent_count += 1  # Increment the agent counter


    # method to reset the simulation
    def reset(self, num_agents):
        # Remove the robots
        for robotId in self.robotIds:
            p.removeBody(robotId)

        # Remove the specimens
        for specimenId in self.specimenIds:
            p.removeBody(specimenId)

        # Create the robots
        self.create_robots(num_agents)

    # method to run the simulation for a specified number of steps
    def run(self, actions, num_steps=1):
        #self.apply_actions(actions)
        start = time.time()
        n = 100
        for i in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()
            time.sleep(1./240.)

            # reset the droplet after 20 steps
            # if self.dropped:
            #     self.cooldown += 1
            #     if self.cooldown == self.cooldown_time:
            #         self.dropped = False
            #         self.cooldown = 0                

            #compute and display the frames per second every n steps
            # if i % n == 0:
            #     fps = n / (time.time() - start)
            #     start = time.time()
            #     print(f'fps: {fps}')
                # #print the orientation of the robot every n steps
                # for i in range(len(self.robotIds)):
                #     orientation = p.getBasePositionAndOrientation(self.robotIds[i])[1]
                #     #print(f'robot {i} orientation: {orientation}')
                #     #get the position of the link on the z axis
                #     link_state = p.getLinkState(self.robotIds[i], 0)
                #     print(f'robot {i} link_state: {link_state}')
        return self.get_states() 
    
    # method to apply actions to the robots using velocity control
    def apply_actions(self, actions): # actions [[x,y,z,drop], [x,y,z,drop], ...
        for i in range(len(self.robotIds)):
            p.setJointMotorControl2(self.robotIds[i], 0, p.VELOCITY_CONTROL, targetVelocity=actions[i][0], force=500)
            p.setJointMotorControl2(self.robotIds[i], 1, p.VELOCITY_CONTROL, targetVelocity=actions[i][1], force=500)
            p.setJointMotorControl2(self.robotIds[i], 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)
            if actions[i][3] == 1:
                self.drop(robotId=self.robotIds[i])
                #logging.info(f'drop: {i}')

    # method to drop a simulated droplet on the specimen from the pipette
    def drop(self, robotId):
        # Get the position of the pipette based on the x,y,z coordinates of the joints
        #robot_position = [0, 0, 0]
        #get the position of the robot
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        robot_position = list(robot_position)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        # x,y offset
        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2]
        # Get the position of the specimen
        specimen_position = p.getBasePositionAndOrientation(self.specimenIds[0])[0]
        #logging.info(f'droplet_position: {droplet_position}')
        # Create a sphere to represent the droplet
        sphereRadius = 0.003  # Adjust as needed
        sphereColor = [1, 0, 0, 0.5]  # RGBA (Red in this case)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=sphereColor)
        #add collision to the sphere
        collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius)
        sphereBody = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=visualShapeId, baseCollisionShapeIndex=collision)
        # Calculate the position of the droplet at the tip of the pipette but at the same z coordinate as the specimen
        droplet_position = [robot_position[0]+x_offset, robot_position[1]+y_offset, robot_position[2]+z_offset]
                            #specimen_position[2] + sphereRadius+0.015/2+0.06]
        p.resetBasePositionAndOrientation(sphereBody, droplet_position, [0, 0, 0, 1])
        self.dropped = True
        #TODO: add some randomness to the droplet position proportional to the height of the pipette above the specimen and the velocity of the pipette of the pipette
        return droplet_position

    # method to get the states of the robots
    def get_states(self):
        states = []
        for robotId in self.robotIds:
            joint_states = p.getJointStates(robotId, [0, 1, 2])
            #logging.info(f'joint_states: {joint_states}')
            states.append(list(joint_states))
            # calculate pipette position
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            robot_position = list(robot_position)
            joint_states = p.getJointStates(robotId, [0, 1, 2])
            robot_position[0] -= joint_states[0][0]
            robot_position[1] -= joint_states[1][0]
            robot_position[2] += joint_states[2][0]
            # pipette position
            pipette_position = [robot_position[0]+self.pipette_offset[0], robot_position[1]+self.pipette_offset[1], robot_position[2]+self.pipette_offset[2]]
            # round to 4 decimal places
            pipette_position = [round(pipette_position[0], 4), round(pipette_position[1], 4), round(pipette_position[2], 4)]
            #logging.info(f'pipette_position: {pipette_position}')
            states[-1].append(pipette_position)

        #logging.info(f'states: {states}')
        return states
    



