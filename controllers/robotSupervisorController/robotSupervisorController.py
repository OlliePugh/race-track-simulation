#!/usr/bin/env python3.10
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalizeToRange, plotData
from PPO_agent import PPOAgent, Transition
import math


from gym.spaces import Box, Discrete
import numpy as np
from shapely.geometry import LineString as shLs, Point as shPt

def get_closest_point_on_track(car_pos, track_line):
    point = shPt(*car_pos)
    nearest_point = track_line.interpolate(track_line.project(point))
    return nearest_point

def get_waypoints(waypoints_parent):
    waypoints = []
    for i in range(waypoints_parent.count):  # for each waypoint
        current_waypoint = waypoints_parent.getMFNode(i)
        waypoints.append(current_waypoint.getField("translation").getSFVec3f()[:-1])  # get the x and y
    
    return shLs([*waypoints, waypoints[0]])  # return a fully connected shapley line
    

class CarRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=np.array([ 0,      0,      0,     0,      0,      0,]),
                                     high=np.array([np.inf, np.inf, np.inf,np.inf, np.inf, np.inf, ]),
                                     dtype=np.float64)
        self.action_space = Discrete(6) # forward, forward left, forward right, backward, backward left, backward right

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        
        self.lidar = self.getDevice("lidar")
        
        self.left_motor = self.getDevice('left_wheel_motor')
        self.right_motor = self.getDevice('right_wheel_motor')
        self.left_servo = self.getDevice('front_left_servo')
        self.right_servo = self.getDevice('front_right_servo')
        
        # Set the target position of the motors
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        self.left_servo.setPosition(float('inf'))
        self.right_servo.setPosition(float('inf'))
        self.lidar.enable(self.timestep)
        
        self.wheels = []
        self.waypoints = get_waypoints(self.getFromDef("waypoints").getField("children"))

        self.stepsPerEpisode = 2000  # Max number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.episodeScore = 0
        self.last_waypoint_score = 0
        self.has_lapped = False

    def get_observations(self):
        range_image = self.lidar.getRangeImage()

        distances = [normalizeToRange(item, self.lidar.getMinRange(), self.lidar.getMaxRange(), 0, 1, clip=True) for item in range_image]
        return distances

    def get_reward(self, action=None):
        point = shPt(*self.robot.getPosition()[:-1])
        
        line_completion = self.waypoints.project(point)
        
        percent_complete = (line_completion / self.waypoints.length) - (self.last_waypoint_score / self.waypoints.length)

        reward = (line_completion - self.last_waypoint_score)*5
        # was at end on last step and is now at start on this step
        is_wrap_around = reward < 0 and self.last_waypoint_score / self.waypoints.length > 0.95 and line_completion / self.waypoints.length < 0.05
        if is_wrap_around:  # completed a lap
            reward = 10
        if reward < 0 and not is_wrap_around:  # the car just went backwards
            reward *= 10  # increase the negative reward
           
        
        self.last_waypoint_score = line_completion
        return reward

    def is_done(self):
        if self.episodeScore > 120.0:
            return True
        
        range_image = self.lidar.getRangeImage()
        if any(value < 0.1 for value in range_image):  # the car is too close to the walls
            return True
            
        return False

    def solved(self):
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 110:  # Last 100 episodes' scores average value
                return True
        return False
        
    def get_default_observation(self):
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def apply_action(self, action): 
        # TODO replace this with an enum
        # 0 = FL
        # 1 = F
        # 2 = FR
        # 3 = BL
        # 4 = B
        # 5 = BR
        
        action = int(action[0])
        match action % 3:
            case 0:  # left
                self.left_servo.setPosition(0.5)
                self.right_servo.setPosition(0.5)
            case 1: # straight
                self.left_servo.setPosition(0)
                self.right_servo.setPosition(0)
            case 2: # right
                self.left_servo.setPosition(-0.5)
                self.right_servo.setPosition(-0.5)
                
        match math.floor(action/3):
            case 0:
                self.left_motor.setPosition(float('inf'))
                self.left_motor.setVelocity(60)
                self.right_motor.setPosition(float('inf'))
                self.right_motor.setVelocity(60)
            case 1:
                self.left_motor.setPosition(float('inf'))
                self.left_motor.setVelocity(-60)
                self.right_motor.setPosition(float('inf'))
                self.right_motor.setVelocity(-60)
        return

    def render(self, mode='human'):
        print("render() is not used")

    def get_info(self):
        return None


env = CarRobot()
agent = PPOAgent(numberOfInputs=env.observation_space.shape[0], numberOfActorOutputs=env.action_space.n, gamma=0.99, use_cuda=False)

def train(agent):
    solved = False
    episodeCount = 0
    episodeLimit = 2000
    # Run outer loop until the episodes limit is reached or the task is solved
    # while not solved and episodeCount < episodeLimit:
    while not solved:  # just keep going forever
        observation = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0
        env.last_waypoint_score = 0
        env.has_lapped = False
        for step in range(env.stepsPerEpisode):
            # In training mode the agent samples from the probability distribution, naturally implementing exploration
            selectedAction, actionProb = agent.work(observation, type_="selectAction")
            # Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached
            # the done condition
            newObservation, reward, done, info = env.step([selectedAction])

            # Save the current state transition in agent's memory
            trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
            agent.storeTransition(trans)
            if done:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                agent.trainStep(batchSize=step + 1)
                solved = env.solved()  # Check whether the task is solved
                break

            if step > env.stepsPerEpisode:
                print(f"step is currently {step} which is higher than steps per episode?!")

            env.episodeScore += reward  # Accumulate episode reward
            observation = newObservation  # observation for next step is current step's newObservation
        print("Episode #", episodeCount, "score:", env.episodeScore)
        episodeCount += 1  # Increment episode counter

    if not solved:
        print("Task is not solved, deploying agent for testing...")
    elif solved:
        print("Task is solved, deploying agent for testing...")
    agent.save("../models/")

observation = env.reset()
agent.load("../models/success")
while True:
    selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
    observation, _, _, _ = env.step([selectedAction])