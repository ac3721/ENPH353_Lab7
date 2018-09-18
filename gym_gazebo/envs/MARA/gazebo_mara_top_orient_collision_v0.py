import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
import copy
import rospkg
import threading # Used for time locks to synchronize position data.

from gazebo_msgs.srv import SpawnModel, DeleteModel

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import WrenchStamped
from gazebo_msgs.msg import ContactState

from sensor_msgs.msg import CompressedImage
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# ROS 2
# import rclpy
# from rclpy.qos import QoSProfile, qos_profile_sensor_data
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing scara joint angles.
# from control_msgs.msg import JointTrajectoryControllerState
# from std_msgs.msg import String

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from baselines.agent.scara_arm.tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

import cv2

import quaternion as quat


# from custom baselines repository
from baselines.agent.utility.general_utils import forward_kinematics, get_ee_points, rotation_from_matrix, \
    get_rotation_matrix,quaternion_from_matrix # For getting points and velocities.

class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """Error object exclusively raised by _process_observations."""
    pass


class GazeboMARATopOrientCollisionv0Env(gazebo_env.GazeboEnv):
    """
    This environment present a modular SCARA robot with a range finder at its
    end pointing towards the workspace of the robot. The goal of this environment is
    defined to reach the center of the "H" or the "O" from the "H-ROS" logo within the worspace.
    This environment uses `slowness=1` and matches the delay between actions/observations
    to this value (slowness). In other words, actions are taken at "1/slowness" rate.

    Reward is determined ... (TODO: describe the heuristic or reward calculation method)
    """
    def __init__(self):
        """
        Initialize the SCARA environemnt
            NOTE: This environment uses ROS and interfaces.

            TODO: port everything to ROS 2 natively
        """
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "MARATop6DOF_Collision_v0.launch")

        # TODO: cleanup this variables, remove the ones that aren't used
        # class variables
        self._observation_msg = None
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None
        self.obs = None
        self.reward = None
        self.done = None
        self.reward_dist = None
        self.reward_ctrl = None
        self.action_space = None
        self.max_episode_steps = 1000 # now used in all algorithms
        self.iterator = 0
        # default to seconds
        self.slowness = 1
        self.slowness_unit = 'sec'
        self.reset_jnts = True
        self._collision_msg = None

        self._time_lock = threading.RLock()

        #############################
        #   Environment hyperparams
        #############################
        # target, where should the agent reach

        EE_POS_TGT = np.asmatrix([-0.40028, 0.095615, 0.72466]) # alex2
        # EE_POS_TGT = np.asmatrix([-0.580238, -0.179591, 0.72466]) # rubik touching the bar
        # EE_ROT_TGT = np.asmatrix([[-0.00128296,  0.9999805 ,  0.00611158],
        #                            [ 0.00231397, -0.0061086 ,  0.99997867],
        #                            [ 0.9999965 ,  0.00129708, -0.00230609]])
        # EE_POS_TGT = np.asmatrix([-0.390768, 0.0101776, 0.725335]) # 200 cm from the z axis
        # EE_POS_TGT = np.asmatrix([0.0, 0.001009, 1.64981])
        # EE_POS_TGT = np.asmatrix([-0.4023037912211465, 0.15501116706606247, 0.7238499613771884]) # 200 cm from the z axis

        # # EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.4868]) # center of the H
        # EE_ROT_TGT = np.asmatrix([[-0.99521107,  0.09689605, -0.01288708],
        #                           [-0.09768035, -0.99077857,  0.09389558],
        #                           [-0.00367013,  0.09470474,  0.99549864]])


        # EE_ROT_TGT = np.asmatrix([[-0.99521107,  0.09689605, -0.01288708],
        #                           [-0.09768035, -0.99077857,  0.09389558],
        #                           [-0.00367013,  0.09470474,  0.99549864]])
        # EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_ROT_TGT = np.asmatrix([[0.79660969, -0.51571238,  0.31536287], [0.51531424,  0.85207952,  0.09171542], [-0.31601302,  0.08944959,  0.94452874]]) # original orientation
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])
        # Initial joint position
        # INITIAL_JOINTS = np.array([0., 0., 1., 0., 1.57, 0.])
        INITIAL_JOINTS = np.array([0., 0., 0., 0., 0., 0.])
        # Used to initialize the robot, #TODO, clarify this more
        # STEP_COUNT = 2  # Typically 100.
        # slowness = 10000000 # 10 ms, where 1 second is real life simulation
        # slowness = 1000000 # 1 ms, where 1 second is real life simulation
        # slowness = 1 # use >10 for running trained network in the simulation
        # slowness = 10 # use >10 for running trained network in the simulation

        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/mara_controller/command'
        JOINT_SUBSCRIBER = '/mara_controller/state'

        # joint names:
        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'
        MOTOR4_JOINT = 'motor4'
        MOTOR5_JOINT = 'motor5'
        MOTOR6_JOINT = 'motor6'

        # Set constants for links
        TABLE = 'table'

        BASE = 'base_link'

        MARA_MOTOR1_LINK = 'motor1_link'
        MARA_MOTOR2_LINK = 'motor2_link'
        MARA_MOTOR3_LINK = 'motor3_link'
        MARA_MOTOR4_LINK = 'motor4_link'
        MARA_MOTOR5_LINK = 'motor5_link'
        MARA_MOTOR6_LINK = 'motor6_link'
        EE_LINK = 'ee_link'


        # EE_LINK = 'ee_link'
        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT,
                       MOTOR4_JOINT, MOTOR5_JOINT, MOTOR6_JOINT]
        LINK_NAMES = [TABLE, BASE, MARA_MOTOR1_LINK, MARA_MOTOR2_LINK,
                            MARA_MOTOR3_LINK, MARA_MOTOR4_LINK,
                            MARA_MOTOR5_LINK, MARA_MOTOR6_LINK,
                      EE_LINK]

        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }
        #############################

        # TODO: fix this and make it relative
        # Set the path of the corresponding URDF file from "assets"
        URDF_PATH = rospkg.RosPack().get_path("mara_description") + "/urdf/mara_demo_camera_top.urdf"

        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
        m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
        m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)
        ee_pos_tgt = EE_POS_TGT
        ee_rot_tgt = EE_ROT_TGT

        # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)
        self.realgoal = ee_tgt
        self.target_orientation = ee_rot_tgt

        self.environment = {
            # rk changed this to for the mlsh
            # 'ee_points_tgt': ee_tgt,
            'ee_points_tgt': self.realgoal,
            'ee_point_tgt_orient': self.target_orientation,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            # 'slowness': slowness,
            'reset_conditions': reset_condition,
            'tree_path': URDF_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
        }

        # self.spec = {'timestep_limit': 5, 'reward_threshold':  950.0,}

        # Subscribe to the appropriate topics, taking into account the particular robot
        # ROS 1 implementation
        self._pub = rospy.Publisher(JOINT_PUBLISHER, JointTrajectory)
        self._sub = rospy.Subscriber(JOINT_SUBSCRIBER, JointTrajectoryControllerState, self.observation_callback)

        self._sub_coll = rospy.Subscriber('/gazebo_contacts',ContactState, self.collision_callback)

        # Initialize a tree structure from the robot urdf.
        #   note that the xacro of the urdf is updated by hand.
        # The urdf must be compiled.
        _, self.ur_tree = treeFromFile(self.environment['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.scara_chain = self.ur_tree.getChain(self.environment['link_names'][0], self.environment['link_names'][-1])
        # print("nr of jnts: ", self.scara_chain.getNrOfJoints())
        # Initialize a KDL Jacobian solver from the chain.
        self.jac_solver = ChainJntToJacSolver(self.scara_chain)
        #print(self.jac_solver)
        self._observations_stale = [False for _ in range(1)]
        #print("after observations stale")
        self._currently_resetting = [False for _ in range(1)]
        self.reset_joint_angles = [None for _ in range(1)]

        # TODO review with Risto, we might need the first observation for calling _step()
        # observation = self.take_observation()
        # assert not done
        # self.obs_dim = observation.size
        """
        obs_dim is defined as:
        num_dof + end_effector_points=3 + end_effector_velocities=3
        end_effector_points and end_effector_velocities is constant and equals 3
        recently also added quaternion to the obs, which has dimension=4
        """
        #
        self.obs_dim = self.scara_chain.getNrOfJoints() + 9#7 #6 hardcode it for now
        # # print(observation, _reward)

        # # Here idially we should find the control range of the robot. Unfortunatelly in ROS/KDL there is nothing like this.
        # # I have tested this with the mujoco enviroment and the output is always same low[-1.,-1.], high[1.,1.]
        # #bounds = self.model.actuator_ctrlrange.copy()
        low = -np.pi * np.ones(self.scara_chain.getNrOfJoints())
        high = np.pi * np.ones(self.scara_chain.getNrOfJoints())
        # low = -np.pi * np.ones(self.scara_chain.getNrOfJoints())
        # high = np.pi * np.ones(self.scara_chain.getNrOfJoints())
        # low = -np.inf * np.ones(self.scara_chain.getNrOfJoints())
        # high = np.inf * np.ones(self.scara_chain.getNrOfJoints())
        # print("Action spaces: ", low, high)
        self.action_space = spaces.Box(low, high)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.add_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.remove_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        model_xml = "<?xml version=\"1.0\"?> \
                    <robot name=\"myfirst\"> \
                      <link name=\"world\"> \
                      </link>\
                      <link name=\"cylinder0\">\
                        <visual>\
                          <geometry>\
                            <sphere radius=\"0.01\"/>\
                          </geometry>\
                          <origin xyz=\"0 0 0\"/>\
                          <material name=\"rojotransparente\">\
                              <ambient>0.5 0.5 1.0 0.1</ambient>\
                              <diffuse>0.5 0.5 1.0 0.1</diffuse>\
                          </material>\
                        </visual>\
                        <inertial>\
                          <mass value=\"5.0\"/>\
                          <inertia ixx=\"1.0\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"1.0\" iyz=\"0.0\" izz=\"1.0\"/>\
                        </inertial>\
                      </link>\
                      <joint name=\"world_to_base\" type=\"fixed\"> \
                        <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\
                        <parent link=\"world\"/>\
                        <child link=\"cylinder0\"/>\
                      </joint>\
                      <gazebo reference=\"cylinder0\">\
                        <material>Gazebo/GreenTransparent</material>\
                      </gazebo>\
                    </robot>"
        robot_namespace = ""
        pose = Pose()
        pose.position.x = EE_POS_TGT[0,0];
        pose.position.y = EE_POS_TGT[0,1];
        pose.position.z = EE_POS_TGT[0,2];

        #Static obstacle (not in original code)
        # pose.position.x = 0.25;#
        # pose.position.y = 0.07;#
        # pose.position.z = 0.0;#

        pose.orientation.x = 0;
        pose.orientation.y= 0;
        pose.orientation.z = 0;
        pose.orientation.w = 0;
        reference_frame = ""
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        self.add_model(model_name="target",
                        model_xml=model_xml,
                        robot_namespace="",
                        initial_pose=pose,
                        reference_frame="")


        # Seed the environment
        # Seed the environment
        self._seed()
    # def collision_callback(self, message):
    #     """
    #     Callback method for the subscriber of Collision data
    #     """
    #
    #
    #     if "puzzle_ball_joints::cubie" not in message.collision1_name and "puzzle_ball_joints::cubie" not in message.collision2_name:
    #         if "robot::motor6_link::motor6_link_fixed_joint_lump__robotiq_arg2f_base_link_collision_1" not in message.collision1_name and  "robot::left_outer_finger::left_outer_finger_collision" not in message.collision2_name:
    #             if "puzzle_ball_joints::cubie" not in message.collision1_name or  "robot::table::table_fixed_joint_lump__mara_work_area_link_collision_4" not in message.collision2_name:
    #                 if message.collision1_name and message.collision2_name:
    #                     self._collision_msg =  message
    #                     # print("self._collision_msg: ", self._collision_msg)

    def collision_callback(self, message):
        """
        Callback method for the subscriber of Collision data
        """
        self._collision_msg = None
        if message.collision1_name is not message.collision2_name:
            if "puzzle_ball_joints::cubie" not in message.collision1_name and "puzzle_ball_joints::cubie" not in message.collision2_name:
                if "puzzle_ball_joints::cubie" not in message.collision1_name or  "robot::table::table_fixed_joint_lump__mara_work_area_link_collision_4" not in message.collision2_name:
                    if "robot::motor6_link::motor6_link_fixed_joint_lump__robotiq_arg2f_base_link_collision_1" not in message.collision1_name and  "robot::left_outer_finger::left_outer_finger_collision" not in message.collision2_name:
                        self._collision_msg =  message

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg =  message

    def init_time(self, slowness =1, slowness_unit='sec', reset_jnts=True):
            self.slowness = slowness
            self.slowness_unit = slowness_unit
            self.reset_jnts = reset_jnts
            print("slowness: ", self.slowness)
            print("slowness_unit: ", self.slowness_unit, "type of variable: ", type(slowness_unit))
            print("reset joints: ", self.reset_jnts, "type of variable: ", type(self.reset_jnts))

    def randomizeTargetPositions(self):
        """
        The goal is to test with randomized positions which range between the boundries of the H-ROS logo
        """
        print("In randomize target positions.")
        EE_POS_TGT_RANDOM1 = np.asmatrix([np.random.uniform(0.2852485,0.3883636), np.random.uniform(-0.1746508,0.1701576), 0.2868]) # boundry box of the first half H-ROS letters with +-0.01 offset
        EE_POS_TGT_RANDOM2 = np.asmatrix([np.random.uniform(0.2852485,0.3883636), np.random.uniform(-0.1746508,0.1701576), 0.2868]) # boundry box of the H-ROS letters with +-0.01 offset
        # EE_POS_TGT_RANDOM1 = np.asmatrix([np.random.uniform(0.2852485, 0.3883636), np.random.uniform(-0.1746508, 0.1701576), 0.3746]) # boundry box of whole box H-ROS letters with +-0.01 offset
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        ee_pos_tgt_random1 = EE_POS_TGT_RANDOM1
        ee_pos_tgt_random2 = EE_POS_TGT_RANDOM2

        # leave rotation target same since in scara we do not have rotation of the end-effector
        ee_rot_tgt = EE_ROT_TGT
        target1 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_random1, ee_rot_tgt).T)
        target2 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_random2, ee_rot_tgt).T)

        # self.realgoal = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_random1, ee_rot_tgt).T)

        self.realgoal = target1 if np.random.uniform() < 0.5 else target2
        print("randomizeTarget realgoal: ", self.realgoal)

    def randomizeTarget(self):
        print("calling randomize target")

        EE_POS_TGT_1 = np.asmatrix([-0.189383, -0.123176, 0.894476]) # point 1
        EE_POS_TGT_2 = np.asmatrix([-0.359236, 0.0297278, 0.760402]) # point 2
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])

        ee_pos_tgt_1 = EE_POS_TGT_1
        ee_pos_tgt_2 = EE_POS_TGT_2

        # leave rotation target same since in scara we do not have rotation of the end-effector
        ee_rot_tgt = EE_ROT_TGT

        # Initialize target end effector position
        # ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

        target1 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_1, ee_rot_tgt).T)
        target2 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_2, ee_rot_tgt).T)


        """
        This is for initial test only, we need to change this in the future to be more realistic.
        E.g. covered target -> go to other target. This could be implemented for example with vision.
        """
        self.realgoal = target1 if np.random.uniform() < 0.5 else target2
        print("randomizeTarget realgoal: ", self.realgoal)

    def randomizeMultipleTargets(self):
        print("calling randomize multiple target")

        EE_POS_TGT_1 = np.asmatrix([0.3325683, 0.0657366, 0.2868]) # center of O
        EE_POS_TGT_2 = np.asmatrix([0.3305805, -0.1326121, 0.2868]) # center of the H
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])

        ee_pos_tgt_1 = EE_POS_TGT_1
        ee_pos_tgt_2 = EE_POS_TGT_2

        # leave rotation target same since in scara we do not have rotation of the end-effector
        ee_rot_tgt = EE_ROT_TGT

        # Initialize target end effector position
        # ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

        target1 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_1, ee_rot_tgt).T)
        target2 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_2, ee_rot_tgt).T)


        """
        This is for initial test only, we need to change this in the future to be more realistic.
        E.g. covered target -> go to other target. This could be implemented for example with vision.
        """
        self.realgoal = target1 if np.random.uniform() < 0.5 else target2
        print("randomizeTarget realgoal: ", self.realgoal)

    def get_trajectory_message(self, action, robot_id=0):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = self.environment['joint_order']
        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        action_float = [float(i) for i in action]
        target.positions = action_float
        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        if (self.slowness_unit == 'sec') or (self.slowness_unit is None):
            target.time_from_start.secs = self.slowness
        elif (self.slowness_unit == 'nsec'):
            target.time_from_start.nsecs = self.slowness
        else:
            print("Unrecognized unit. Please use sec or nsec.")

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]
        return action_msg

    def process_observations(self, message, agent, robot_id=0):
        """
        Helper fuinction to convert a ROS message to joint angles and velocities.
        Check for and handle the case where a message is either malformed
        or contains joint values in an order different from that expected observation_callback
        in hyperparams['joint_order']
        """
        if not message:
            print("Message is empty");
            # return None
        else:
            # # Check if joint values are in the expected order and size.
            if message.joint_names != agent['joint_order']:
                # Check that the message is of same size as the expected message.
                if len(message.joint_names) != len(agent['joint_order']):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER

                # Check that all the expected joint values are present in a message.
                if not all(map(lambda x,y: x in y, message.joint_names,
                    [self._valid_joint_set[robot_id] for _ in range(len(message.joint_names))])):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER
                    print("Joints differ")
            return np.array(message.actual.positions) # + message.actual.velocities

    def get_jacobians(self, state, robot_id=0):
        """
        Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
        This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
        The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
        Returns a repackaged Jacobian that is 3x6.
        """
        # Initialize a Jacobian for self.scara_chain.getNrOfJoints() joint angles by 3 cartesian coords and 3 orientation angles
        jacobian = Jacobian(self.scara_chain.getNrOfJoints())
        # Initialize a joint array for the present self.scara_chain.getNrOfJoints() joint angles.
        angles = JntArray(self.scara_chain.getNrOfJoints())
        # Construct the joint array from the most recent joint angles.
        for i in range(self.scara_chain.getNrOfJoints()):
            angles[i] = state[i]
        # Update the jacobian by solving for the given angles.observation_callback
        self.jac_solver.JntToJac(angles, jacobian)
        # Initialize a numpy array to store the Jacobian.
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])
        # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
        ee_jacobians = J
        return ee_jacobians

    def get_ee_points_jacobians(self, ref_jacobian, ee_points, ref_rot):
        """
        Get the jacobians of the points on a link given the jacobian for that link's origin
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :return: 3N x 6 Jac_trans, each 3 x 6 numpy array is the Jacobian[:3, :] for that point
                 3N x 6 Jac_rot, each 3 x 6 numpy array is the Jacobian[3:, :] for that point
        """
        ee_points = np.asarray(ee_points)
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        end_effector_points_rot = np.expand_dims(ref_rot.dot(ee_points.T).T, axis=1)
        ee_points_jac_trans = np.tile(ref_jacobians_trans, (ee_points.shape[0], 1)) + \
                                        np.cross(ref_jacobians_rot.T, end_effector_points_rot).transpose(
                                            (0, 2, 1)).reshape(-1, self.scara_chain.getNrOfJoints())
        ee_points_jac_rot = np.tile(ref_jacobians_rot, (ee_points.shape[0], 1))
        return ee_points_jac_trans, ee_points_jac_rot

    def get_ee_points_velocities(self, ref_jacobian, ee_points, ref_rot, joint_velocities):
        """
        Get the velocities of the points on a link
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :param joint_velocities: 1 x 6 numpy array, joint velocities
        :return: 3N numpy array, velocities of each point
        """
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        ee_velocities_trans = np.dot(ref_jacobians_trans, joint_velocities)
        ee_velocities_rot = np.dot(ref_jacobians_rot, joint_velocities)
        ee_velocities = ee_velocities_trans + np.cross(ee_velocities_rot.reshape(1, 3),
                                                       ref_rot.dot(ee_points.T).T)
        return ee_velocities.reshape(-1)

    def take_observation(self):
        """
        Take observation from the environment and return it.
        TODO: define return type
        """
        # Take an observation
        # done = False

        obs_message = self._observation_msg
        if obs_message is None:
            # print("last_observations is empty")
            return None

        # Collect the end effector points and velocities in
        # cartesian coordinates for the process_observationsstate.
        # Collect the present joint angles and velocities from ROS for the state.
        last_observations = self.process_observations(obs_message, self.environment)
        # # # Get Jacobians from present joint angles and KDL trees
        # # # The Jacobians consist of a 6x6 matrix getting its from from
        # # # (# joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
        ee_link_jacobians = self.get_jacobians(last_observations)
        if self.environment['link_names'][-1] is None:
            print("End link is empty!!")
            return None
        else:
            # print(self.environment['link_names'][-1])
            trans, rot = forward_kinematics(self.scara_chain,
                                        self.environment['link_names'],
                                        last_observations[:self.scara_chain.getNrOfJoints()],
                                        base_link=self.environment['link_names'][0],
                                        end_link=self.environment['link_names'][-1])
            # #
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = rot
            rotation_matrix[:3, 3] = trans
            # angle, dir, _ = rotation_from_matrix(rotation_matrix)
            # #
            # current_quaternion = np.array([angle]+dir.tolist())#

            # # I need this calculations for the new reward function, need to send them back to the run mara or calculate them here
            # current_quaternion = quaternion_from_matrix(rotation_matrix)
            # tgt_quartenion = quaternion_from_matrix(self.target_orientation)

            current_quaternion = quat.from_rotation_matrix(rotation_matrix)
            # print("current_quaternion: ", current_quaternion)
            tgt_quartenion = quat.from_rotation_matrix(self.target_orientation)

            # A  = np.vstack([current_quaternion, np.ones(len(current_quaternion))]).T

            #quat_error = np.linalg.lstsq(A, tgt_quartenion)[0]

            # this is wrong!!!! Substraction should be replaced by: quat_error = current_quaternion * tgt_quartenion.conjugate()
            # quat_error = current_quaternion - tgt_quartenion

            quat_error = current_quaternion * tgt_quartenion.conjugate()
            rot_vec_err = quat.as_rotation_vector(quat_error)

            # convert quat to np arrays
            quat_error = quat.as_float_array(quat_error)



            # RK:  revisit this later, we only take one angle difference here!
            angle_diff = 2 * np.arccos(np.clip(quat_error[..., 0], -1., 1.))
            # print("quat error: ", quat_error)
            # print("quat_error[..., 0]: ", quat_error[..., 0])
            # print("quat_error: ",quat_error)
            # print("angle_diff: ", angle_diff)
            # print("self.realgoal: ", self.realgoal)
            # print("curr quat: ", current_quaternion)
            current_ee_tgt = np.ndarray.flatten(get_ee_points(self.environment['end_effector_points'],
                                                              trans,
                                                              rot).T)
            ee_points = current_ee_tgt - self.realgoal#self.environment['ee_points_tgt']
            ee_points_jac_trans, _ = self.get_ee_points_jacobians(ee_link_jacobians,
                                                                   self.environment['end_effector_points'],
                                                                   rot)
            ee_velocities = self.get_ee_points_velocities(ee_link_jacobians,
                                                           self.environment['end_effector_points'],
                                                           rot,
                                                           last_observations)

            # Concatenate the information that defines the robot state
            # vector, typically denoted asrobot_id 'x'.
            state = np.r_[np.reshape(last_observations, -1),
                          np.reshape(ee_points, -1),
                          np.reshape(rot_vec_err, -1),
                          np.reshape(ee_velocities, -1),]
            # print("quat_error: ", quat_error)
            # print("ee_points:", ee_points)
            # print("angle_diff: ", angle_diff)
            return np.r_[np.reshape(last_observations, -1),
                          np.reshape(ee_points, -1),
                          np.reshape(rot_vec_err, -1),
                          np.reshape(ee_velocities, -1),]

    def rmse_func(self, ee_points):
        """
        Computes the Residual Mean Square Error of the difference between current and desired end-effector position
        """
        rmse = np.sqrt(np.mean(np.square(ee_points), dtype=np.float32))
        return rmse

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action): # , prevac
        """
        Implement the environment step abstraction. Execute action and returns:
            - reward
            - done (status)
            - action
            - observation
            - dictionary (#TODO clarify)
        """
        self.iterator+=1
        # rmse_trans = self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])
        # rmse_orient = self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+7)])
        # # print("rmse_orient: ", self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+7)])
        # self.reward_dist = -rmse_trans
        # self.reward_orient = -rmse_orient
        #                     # + self.ob[(self.scara_chain.getNrOfJoints()+4)] )
        #
        # # here we want to fetch the positions of the end-effector which are nr_dof:nr_dof+3
        # if(self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])<0.005):
        #     self.reward_final_dist = 1 + self.reward_dist # Make the reward increase as the distance decreases
        #     print("Reward Pose is: ", self.reward_final_dist)
        # else:
        #     self.reward_final_dist = self.reward_dist

        self.reward_dist = -self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])
        # careful we have degrees now so we scale with
        orientation_scale = 0.01

        # print("orientation reward: ", self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+4)])
        self.reward_orient = - orientation_scale * self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+6)])#self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+4)])*0.1
        # print("self.reward_orient: ", self.reward_orient)
        # print(self.reward_orient)
        # print("self.reward_orient: ", self.reward_orient)

        #scale here the orientation because it should not be the main bias of the reward, position should be
        collided = False

        if self._collision_msg is not None and self._collision_msg.collision1_name and self._collision_msg.collision2_name:
            # print("\ncollision detected: ", self._collision_msg)
            print("Collision detected")
            collided = True
            self.reward = (self.reward_dist + self.reward_orient) * 4.0
            # print("Reward collision is: ", self.reward)

            # Resets the state of the environment and returns an initial observation.
            # we should avoid this --> huge bottleneck
            rospy.wait_for_service('/gazebo/reset_simulation')
            try:
                self.reset_proxy()
                # go to the previous state before colliding
                #self._pub.publish(self.get_trajectory_message(prevac[:self.scara_chain.getNrOfJoints()]))
            except (rospy.ServiceException) as e:
                print ("/gazebo/reset_simulation service call failed")
                # self.goToInit()
                # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        else:
            # here we want to fetch the positions of the end-effector which are nr_dof:nr_dof+3
            # here is the distance block
            if(self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])<0.01):
                self.reward = 1 + self.reward_dist # Make the reward increase as the distance decreases
                print("Reward is: ", self.reward)
            else:
                self.reward = self.reward_dist
                # print("Reward is (minus): ", self.reward)

            # take into account the orientation
            if(abs(self.reward_orient) < 0.005):
                self.reward = self.reward + (1 + self.reward_orient)
                print("Reward orientation is: ", self.reward)
            else:
                self.reward = self.reward + self.reward_orient# * self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+7)])
                # print("Reward orientation is (minus): ", self.reward)

            if  self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])<0.01 and abs(self.reward_orient)<0.005:
                self.reward = 10 * (2 + self.reward_dist + self.reward_orient)
                print("Reward hit the target, and is: ", self.reward)

        # self.reward =self.reward - abs(self.ob[(self.scara_chain.getNrOfJoints()+4)])
        # Calculate if the env has been solved
        done = bool(((abs(self.reward_dist) < 0.005) and (abs(self.reward_orient)) < 0.005) or (self.iterator>self.max_episode_steps))

        # Execute "action"
        self._pub.publish(self.get_trajectory_message(action[:self.scara_chain.getNrOfJoints()]))

        # # Take an observation
        # TODO: program this better, check that ob is not None, etc.
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Return the corresponding observations, rewards, etc.
        # TODO, understand better what's the last object to return
        return self.ob, self.reward, done, {}


    def goToInit(self):
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()
        # # Go to initial position and wait until it arrives there
        # Wait until the arm is within epsilon of reset configuration.
        self._time_lock.acquire(True, -1)
        with self._time_lock:
            self._currently_resetting = True
        self._time_lock.release()

        if self._currently_resetting:
            epsilon = 1e-3
            reset_action = self.environment['reset_conditions']['initial_positions']
            now_action = self._observation_msg.actual.positions
            du = np.linalg.norm(reset_action-now_action, float(np.inf))
            self._pub.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))
            if du > epsilon:
                self._currently_resetting = True
                self._pub.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))
                time.sleep(3)

    def _reset(self):
        """
        Reset the agent for a particular experiment condition.
        """

        self.iterator = 0

        if self.reset_jnts is True:
            self._pub.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))
            if (self.slowness_unit == 'sec') or (self.slowness_unit is None):
                time.sleep(int(self.slowness))
            elif(self.slowness_unit == 'nsec'):
                time.sleep(int(self.slowness/1000000000)) # using nanoseconds
            else:
                print("Unrecognized unit. Please use sec or nsec.")

        # Take an observation
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Return the corresponding observation
        return self.ob
