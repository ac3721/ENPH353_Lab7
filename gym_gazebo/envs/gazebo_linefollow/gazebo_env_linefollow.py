
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


     #Get first value of road
    def first_val(self,row):
        for i in range(len(row)):
            if row[i] == 0:
                return i
        return 0

    #Get last value of road
    def last_val(self,row):
        state = 0
        for i in range(len(row)):
            if state == 0:
                if row[i] == 0:
                    state=1
            elif state == 1:
                if row[i] != 0:
                    return (i-1)
        return len(row)-1 


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='mono8')
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        height, width = cv_image.shape
        #pick bottom ish of screen
        height = int(height*0.95)
        increment = int(width/10)

        #Threshold then binary
        _, thresholded = cv2.threshold(cv_image,125, 255, cv2.THRESH_BINARY)

        #Convert image into NumPy array
        arr = np.array(thresholded)

        #Want binary array
        bin = (arr > 128).astype(int)
        
        #Get Road
        row = bin [height, :]
        x_val = []
        x_val.append(self.first_val(row))
        x_val.append(self.last_val(row))
        x_coord = int(np.mean(x_val)) #x coord of center of road

        row1 = bin [height+1, :]
        x_val1 = []
        x_val1.append(self.first_val(row1))
        x_val1.append(self.last_val(row1))
        x_coord1 = int(np.mean(x_val1)) 

        row2 = bin [height-1, :]
        x_val2 = []
        x_val2.append(self.first_val(row2))
        x_val2.append(self.last_val(row2))
        x_coord2 = int(np.mean(x_val2)) 

        #For stability, take average of 3 row to get centroid
        x_o = []
        status = False
        if not(self.first_val(row) == 0 and self.last_val(row) == width-1):
            x_o.append(int(np.mean(x_val)))
            status = True
        if not(self.first_val(row1) == 0 and self.last_val(row1) == width-1):
            x_o.append(int(np.mean(x_val1)))
            status = True
        if not(self.first_val(row2) == 0 and self.last_val(row2) == width-1):
            x_o.append(int(np.mean(x_val2)))
            status = True
        # if ((self.first_val(row) == 0 and self.last_val(row) == width-1) and (self.first_val(row1) == 0 and self.last_val(row1) == width) and (self.first_val(row2) == 0 and self.last_val(row2) == width)):
        #     status = False

        if (status == False):
            if (self.timeout == 30):
                done = True
            else:
                self.timeout +=1

        #print (width, self.first_val(row),self.last_val(row),self.first_val(row1),self.last_val(row1))

        if (status == True):
            x = int(np.mean(x_o)) 
            if (x < increment):
                state = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif (x < 2 * increment):
                state = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif (x < 3 * increment):
                state = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            elif (x < 4 * increment):
                state = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif (x < 5 * increment):
                state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif (x < 6 * increment):
                state = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif (x < 7 * increment):
                state = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif (x < 8 * increment):
                state = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            elif (x < 9 * increment):
                state = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif (x < width):
                state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

            if (done == False):
                cv2.circle(cv_image, (x, height), 1, (0, 0, 255), 30)


        cv2.imshow("raw", cv_image)
        cv2.waitKey(100)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.6
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
