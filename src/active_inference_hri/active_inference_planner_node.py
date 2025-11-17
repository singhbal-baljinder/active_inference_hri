#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR self.A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
import message_filters

from geometry_msgs.msg import Pose, PoseStamped, WrenchStamped
from std_msgs.msg import Float64
from active_inference_planner.msg import StateInference
import numpy as np

from pymdp.agent import Agent
from pymdp import utils
import copy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image




# Assign the indices names so that when we build the self.A matrices, things will be more 'semantically' obvious
# States names
GOAL_IS_G1, GOAL_IS_G2 = 0, 1 
NO_MOTION, SMALL_MOTION_TO_G1, SMALL_MOTION_TO_G2, GO_TO_G1, GO_TO_G2 = 0, 1, 2, 3, 4
NOT_PROACTIVE_STATE, PROACTIVE_STATE = 0, 1 
# Observations names
AROUND_ZERO_FORCE_EVIDENCE, TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE, TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE, NOT_SMOOTH_FORCE_EVIDENCE, SMOOTH_FORCE_EVIDENCE = 0, 1, 2, 3, 4
# Self observations names
NO_MOTION_O, SMALL_MOTION_TO_G1_0, SMALL_MOTION_TO_G2_0, GO_TO_G1_0, GO_TO_G2_0 = 0, 1, 2, 3, 4
# Actions names
NULL_ACTION = 0
NO_MOTION_ACTION, SMALL_MOTION_TO_G1_ACTION, SMALL_MOTION_TO_G2_ACTION, GO_TO_G1_ACTION, GO_TO_G2_ACTION = 0, 1, 2, 3, 4

CLOSE_TO_GOAL_DISTANCE = 0.01
CLOSE_TO_SETPOINT_DISTANCE = 0.002

class ActiveInferencePlanner:

    pub_topic_name = "active_inference/pose_set_point"
    pub_topic_name_stiffness = "active_inference/stiffness_set_point"
    pub_beleif_topic_name = "active_inference/belief"
    pub_video_topic_name = "active_inference/webcam_video"
    # Name of the topic to subscribe
    sub_wrench_topic_name = "franka_state_controller/F_ext"
    sub_ee_pose_topic_name = "my_cartesian_impedance_controller/curr_ee_pose"
    sub_ee_cmd_pose_topic_name = "my_cartesian_impedance_controller/equilibrium_pose"

    set_point = PoseStamped()

    # The number of samples to consider in the Interaction Window
    T_inference_win = 10
    # Counter to publish only once every T_inference_win
    counter = 0
    # The external WrenchStamped received in the emotion window of length T_inference_window
    f_ext_win = [0.0 for i in range(T_inference_win)]
    # The wrist absolute velocity received in the emotion window of length T_inference_window
    ee_pose_win = [0.0 for i in range(T_inference_win)]
    # The external force filtered with a high pass filter
    f_ext_gradient_win = np.array([0.0 for i in range(T_inference_win)])
    f_ext_gradient_max = 0.0
    # True goal state
    true_goal_state = GOAL_IS_G2
    # Position of hte goals in the robot base frame
    goal_positions = [0.15, 0.45]
    explorative_dev = 0.05
    # In Newtons
    initial_loop_done = False
    initial_force_bias = 0.0
    high_force_threshold = 2.0
    f_ext_gradient_thresh = 2.0
    risk_parameter = 1
    avg_force = 0.0
    f_ext_gradient_max = 0.0

    K_min = 200
    K_max = 700
    K_0 = 0
    stiffness_desired = K_max

    ee_pose_cmd_x = 0.0
    belief_state = [np.array([0,0]), np.array([1,0,0,0,0]), np.array([0,0])]

    log_more_info = False
    
    T_init_loop_length = 100
    counter_init = 0

    
    # Create a CvBridge to convert OpenCV images to ROS Image messages
    bridge = CvBridge()
    cap = cv2.VideoCapture(2)

    def __init__(self) -> None:
        self.factor_names =  ["GOAL_STATE", "CHOICE_STATE", "PROACTIVE_STATE"]
        # The total number of hidden state factors
        self.num_factors = len(self.factor_names) 

        # The list of the dimensionalities of each hidden state factor 
        self.num_states = [len([GOAL_IS_G1,GOAL_IS_G2]), len([NO_MOTION, SMALL_MOTION_TO_G1, SMALL_MOTION_TO_G2, GO_TO_G1, GO_TO_G2]), len([NOT_PROACTIVE_STATE, PROACTIVE_STATE])] # this is a list of the dimensionalities of each hidden state factor 
        self.num_obs = [len([AROUND_ZERO_FORCE_EVIDENCE, TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE, TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE, NOT_SMOOTH_FORCE_EVIDENCE, SMOOTH_FORCE_EVIDENCE]),
                                   len([NO_MOTION_O, SMALL_MOTION_TO_G1_0, SMALL_MOTION_TO_G2_0, GO_TO_G1_0, GO_TO_G2_0])]

        self.modality_names = ["FORCE_OBS", "ACTION_SELF_OBS"]
        self.num_modalities = len(self.modality_names)
        self.A = utils.obj_array_zeros([[o] + self.num_states for _, o in enumerate(self.num_obs)])

        # Note 1 : if we want to embed the idea that "sometimes if you are not moving and the goal is at the left you will not receive zero force" we can play
        #          with the probabilities self.A[0][AROUND_ZERO_FORCE_EVIDENCE, 0,  NO_MOTION]  and self.A[0][AROUND_ZERO_FORCE_EVIDENCE, 1,  NO_MOTION]  (one per column, i.e. on per goal location).
        # Note 2: if someone is hesitant she will have lower probability of signaling her intention through force when the robot is not moving but might react properly if the robot moves.
        #         This could be taken into account by taking high probability for the row self.A[0][AROUND_ZERO_FORCE_EVIDENCE,:,  NO_MOTION] for hesitant people and low for determined people.
        # Note 3: an alternative way to account for different human behaviors is to consider them as states. For example we could have hesitant_interaction/determined_interaction 
        #         states and do inference also on them.
        p = 0.6
        q = (1 - p)/2

        if (p + q) != 1:
            ValueError("The columns of A must sum to 1")

        self.A[0][AROUND_ZERO_FORCE_EVIDENCE,:,  NO_MOTION, NOT_PROACTIVE_STATE] = p # they always get the 'around zero force' observation in the NO_MOTION state
        self.A[0][TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE,:,  NO_MOTION, NOT_PROACTIVE_STATE] = q 
        self.A[0][TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE,:,  NO_MOTION, NOT_PROACTIVE_STATE] = q
        # Explantion of the first row (similar for the others):
        #    the agent expects to see the NEGATIVE_HIGH_FORCE_EVIDENCE observation with 10% probability, if the GOAL_STATE is GOAL_IS_G1, and the agent is in the MOVE_LEFT state
        self.A[0][AROUND_ZERO_FORCE_EVIDENCE, GOAL_IS_G1, SMALL_MOTION_TO_G1, NOT_PROACTIVE_STATE] = p
        self.A[0][AROUND_ZERO_FORCE_EVIDENCE, GOAL_IS_G2, SMALL_MOTION_TO_G1, NOT_PROACTIVE_STATE] = q
        self.A[0][TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE, :, SMALL_MOTION_TO_G1, NOT_PROACTIVE_STATE] = q
        self.A[0][TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE, GOAL_IS_G1, SMALL_MOTION_TO_G1, NOT_PROACTIVE_STATE] = q
        self.A[0][TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE, GOAL_IS_G2, SMALL_MOTION_TO_G1, NOT_PROACTIVE_STATE] = p


        self.A[0][AROUND_ZERO_FORCE_EVIDENCE, GOAL_IS_G1, SMALL_MOTION_TO_G2, NOT_PROACTIVE_STATE] = q
        self.A[0][AROUND_ZERO_FORCE_EVIDENCE, GOAL_IS_G2, SMALL_MOTION_TO_G2, NOT_PROACTIVE_STATE] = p
        self.A[0][TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE, GOAL_IS_G1, SMALL_MOTION_TO_G2, NOT_PROACTIVE_STATE] = p
        self.A[0][TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE, GOAL_IS_G2, SMALL_MOTION_TO_G2, NOT_PROACTIVE_STATE] = q
        self.A[0][TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE, :, SMALL_MOTION_TO_G2, NOT_PROACTIVE_STATE] = q

        # PROACTIVE_STATE: robot always get the 'around zero force' observation in the NO_MOTION state, and the person will push towards her prefered goal no matter the robot motion state
        p_proactive = 0.9
        q_proactive = (1 - p_proactive)/2

        self.A[0][AROUND_ZERO_FORCE_EVIDENCE,:, :, PROACTIVE_STATE] = q_proactive
        self.A[0][TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE,GOAL_IS_G1, :, PROACTIVE_STATE] = p_proactive
        self.A[0][TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE,GOAL_IS_G2, :, PROACTIVE_STATE] = q_proactive
        self.A[0][TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE,GOAL_IS_G1, :, PROACTIVE_STATE] = q_proactive
        self.A[0][TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE,GOAL_IS_G2, :, PROACTIVE_STATE] = p_proactive


        self.A[0][:,:, GO_TO_G1, PROACTIVE_STATE] = 0
        self.A[0][:,:, GO_TO_G2, PROACTIVE_STATE] = 0

        p_smoothness_proactive = 1.0

        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G1, PROACTIVE_STATE] = 1 - p_smoothness_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G1, PROACTIVE_STATE] = p_smoothness_proactive
        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G1, PROACTIVE_STATE] = p_smoothness_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G1, PROACTIVE_STATE] = 1 - p_smoothness_proactive

        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G2, PROACTIVE_STATE] = p_smoothness_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G2, PROACTIVE_STATE] = 1 - p_smoothness_proactive
        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G2, PROACTIVE_STATE] = 1 - p_smoothness_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G2, PROACTIVE_STATE] = p_smoothness_proactive


        p_smoothness_not_proactive = 0.9

        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G1, NOT_PROACTIVE_STATE] = 1 - p_smoothness_not_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G1, NOT_PROACTIVE_STATE] = p_smoothness_not_proactive
        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G1, NOT_PROACTIVE_STATE] = p_smoothness_not_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G1, NOT_PROACTIVE_STATE] = 1 - p_smoothness_not_proactive

        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G2, NOT_PROACTIVE_STATE] = p_smoothness_not_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G1, GO_TO_G2, NOT_PROACTIVE_STATE] = 1 - p_smoothness_not_proactive
        self.A[0][NOT_SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G2, NOT_PROACTIVE_STATE] = 1 - p_smoothness_not_proactive
        self.A[0][SMOOTH_FORCE_EVIDENCE, GOAL_IS_G2, GO_TO_G2, NOT_PROACTIVE_STATE] = p_smoothness_not_proactive


        # Modality corresponding to action self observation
        self.A[1][NO_MOTION_O,:,NO_MOTION, :] = 1.0
        self.A[1][SMALL_MOTION_TO_G1_0,:,SMALL_MOTION_TO_G1, :] = 1.0
        self.A[1][SMALL_MOTION_TO_G2_0,:,SMALL_MOTION_TO_G2, :] = 1.0
        self.A[1][GO_TO_G1_0,:,GO_TO_G1, :] = 1.0
        self.A[1][GO_TO_G2_0,:,GO_TO_G2_0, :] = 1.0

        self.control_names = ["NULL_STATE_CONTROL", "CHOICE_STATE_CONTROL", "NULL_STATE_CONTROL_HUMAN"]
        self.num_control_factors = len(self.control_names) # this is the total number of controllable hidden state factors  
        self.num_control = [len([NULL_ACTION]), len([NO_MOTION_ACTION, SMALL_MOTION_TO_G1_ACTION, SMALL_MOTION_TO_G2_ACTION, GO_TO_G1_ACTION, GO_TO_G2_ACTION]), len([NULL_ACTION])] 
        self.control_fac_idx = [1] # this is the (non-trivial) controllable factor, where there will be a >1-dimensional control state along this factor
        self.B = utils.obj_array(self.num_factors)

        p_stoch = 0.0

        # we cannot influence factor zero, set up the 'default' stationary dynamics - 
        # one state just maps to itself at the next timestep with very high probability, by default. So this means the goal state can
        # change from one to another with some low probability (p_stoch)

        self.B[0] = np.zeros((self.num_states[0], self.num_states[0], self.num_control[0])) 
        self.B[0][GOAL_IS_G1, GOAL_IS_G1, NULL_ACTION] = 1.0 - p_stoch
        self.B[0][GOAL_IS_G2, GOAL_IS_G1, NULL_ACTION] = p_stoch

        self.B[0][GOAL_IS_G2, GOAL_IS_G2, NULL_ACTION] = 1.0 - p_stoch
        self.B[0][GOAL_IS_G1, GOAL_IS_G2, NULL_ACTION] = p_stoch

        # setup our controllable factor.
        self.B[1] = np.zeros((self.num_states[1], self.num_states[1], self.num_control[1]))
        self.B[1][NO_MOTION, :, NO_MOTION_ACTION] = 1.0 
        self.B[1][SMALL_MOTION_TO_G1, NO_MOTION, SMALL_MOTION_TO_G1_ACTION] = 1.0
        self.B[1][SMALL_MOTION_TO_G1, SMALL_MOTION_TO_G1, SMALL_MOTION_TO_G1_ACTION] = 1.0
        self.B[1][SMALL_MOTION_TO_G1, SMALL_MOTION_TO_G2, SMALL_MOTION_TO_G1_ACTION] = 1.0
        self.B[1][GO_TO_G1, GO_TO_G1, SMALL_MOTION_TO_G1_ACTION] = 1.0
        self.B[1][GO_TO_G2, GO_TO_G2, SMALL_MOTION_TO_G1_ACTION] = 1.0

        self.B[1][SMALL_MOTION_TO_G2, NO_MOTION, SMALL_MOTION_TO_G2_ACTION] = 1.0
        self.B[1][SMALL_MOTION_TO_G2, SMALL_MOTION_TO_G1, SMALL_MOTION_TO_G2_ACTION] = 1.0
        self.B[1][SMALL_MOTION_TO_G2, SMALL_MOTION_TO_G2, SMALL_MOTION_TO_G2_ACTION] = 1.0
        self.B[1][GO_TO_G1, GO_TO_G1, SMALL_MOTION_TO_G2_ACTION] = 1.0
        self.B[1][GO_TO_G2, GO_TO_G2, SMALL_MOTION_TO_G2_ACTION] = 1.0

        self.B[1][GO_TO_G1, :, GO_TO_G1_ACTION] = 1.0
        self.B[1][GO_TO_G1, GO_TO_G2, GO_TO_G1_ACTION] = 0.0
        self.B[1][GO_TO_G2, GO_TO_G2, GO_TO_G1_ACTION] = 1.0

        self.B[1][GO_TO_G2, :, GO_TO_G2_ACTION] = 1.0
        self.B[1][GO_TO_G2, GO_TO_G1, GO_TO_G2_ACTION] = 0.0
        self.B[1][GO_TO_G1, GO_TO_G1, GO_TO_G2_ACTION] = 1.0

        # setup our controllable factor.
        self.B[2] = np.zeros((self.num_states[2], self.num_states[2], self.num_control[2]))
        p_stoch_proactive = 0.0
        self.B[2][PROACTIVE_STATE, PROACTIVE_STATE, NULL_ACTION] = 1.0 - p_stoch_proactive
        self.B[2][PROACTIVE_STATE, NOT_PROACTIVE_STATE, NULL_ACTION] = p_stoch_proactive

        self.B[2][NOT_PROACTIVE_STATE, NOT_PROACTIVE_STATE, NULL_ACTION] = 1 - p_stoch_proactive
        self.B[2][NOT_PROACTIVE_STATE, PROACTIVE_STATE, NULL_ACTION] = p_stoch_proactive

        # Higher the risk constant the more the agent will take the risk of selecting right away to go to a particular goal
        self.risk_parameter = 3
        self.C = utils.obj_array_zeros([num_ob for num_ob in self.num_obs])
        self.C[0][AROUND_ZERO_FORCE_EVIDENCE] = 0.0  # make the observation we've a priori named `AROUND_ZERO_FORCE_EVIDENCE` actually desirable, by building a high prior expectation of encountering it 
        self.C[0][TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE] = 0.0    # make the observation we've a prior named `TOWARDS_LEFT_HIGH_FORCE_EVIDENCE` actually aversive,by building a low prior expectation of encountering it
        self.C[0][TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE] = 0.0 # make the observation we've a prior named `TOWARDS_RIGHT_HIGH_FORCE_EVIDENCE` actually aversive,by building a low prior expectation of encountering it
        self.C[0][NOT_SMOOTH_FORCE_EVIDENCE] = -5*self.risk_parameter
        self.C[0][SMOOTH_FORCE_EVIDENCE] = self.risk_parameter
        
        # The initial belief on the state factors
        self.D = utils.obj_array(self.num_factors)
        self.D[0] = np.zeros((self.num_states[0]))
        self.D[0][0] = 0.5
        self.D[0][1] = 0.5
        self.D[1] = np.zeros((self.num_states[1]))
        self.D[1][0] = 1
        self.D[1][1] = 0
        self.D[1][2] = 0
        self.D[1][3] = 0
        self.D[1][4] = 0
        self.D[2] = np.zeros((self.num_states[2]))
        self.D[2][0] = 1/2
        self.D[2][1] = 1/2

        self.agent = Agent(A=self.A, B=self.B, C=self.C, D=self.D, control_fac_idx=self.control_fac_idx)

        # transition/observation matrices characterising the generative process
        self.A_gp = copy.deepcopy(self.A)
        self.B_gp = copy.deepcopy(self.B)

        # Init belief
        self.belief_state = [np.array([self.D[0][0], self.D[0][1]]),
                             np.array([self.D[1][0], self.D[1][1], self.D[1][2], self.D[1][3], self.D[1][4]]),
                             np.array([self.D[2][0], self.D[2][1]])]
        # Initial observation
        self.observation = [AROUND_ZERO_FORCE_EVIDENCE, NO_MOTION_O] 
        # Initial (true) state, sometimes refered as D vector
        self.state = [self.true_goal_state, NO_MOTION, NOT_PROACTIVE_STATE] 

        self.states_idx_names = [ ["GOAL_1", "GOAL_2"], 
                 ["NO_MOTION", "SMALL_MOTION_TO_G1", "SMALL_MOTION_TO_G2", "GO_TO_G1", "GO_TO_G2"],
                 ["NOT_PROACTIVE", "PROACTIVE"]]

        self.obs_idx_names = [ ["AROUND_ZERO_FORCE_EVIDENCE", "TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE", "TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE","NOT_SMOOTH_FORCE_EVIDENCE", "SMOOTH_FORCE_EVIDENCE"],
                ["NO_MOTION", "SMALL_MOTION_TO_G1", "SMALL_MOTION_TO_G2", "GO_TO_G1", "GO_TO_G2"] ]

        self.action_idx_names = [ ["NULL"], ["NO_MOTION", "SMALL_MOTION_TO_G1", "SMALL_MOTION_TO_G2", "GO_TO_G1", "GO_TO_G2"], ["NULL"]]

        ######### ROS initialization ###########
        self.pub = rospy.Publisher(
            self.pub_topic_name, PoseStamped, queue_size=10
        )

        self.belief_publisher = rospy.Publisher(self.pub_beleif_topic_name, StateInference, queue_size=10)
        self.belief_msg = StateInference()
        self.belief_msg.state_names = self.factor_names
        
        self.pub_stiffness = rospy.Publisher(    
            self.pub_topic_name_stiffness, Float64, queue_size=10
        )
        # Use ROS tools for message synchronization
        self.sub_f_ext = message_filters.Subscriber(
            self.sub_wrench_topic_name, WrenchStamped
        )
        self.sub_ee_pose = message_filters.Subscriber(
            self.sub_ee_pose_topic_name, Pose
        )
        self.sub_ee_pose_cmd = rospy.Subscriber(self.sub_ee_cmd_pose_topic_name, PoseStamped, self.subscriber_callbak_ee_pose_cmd)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_f_ext, self.sub_ee_pose],
            10,
            0.01,
            allow_headerless=True,
        )
        ts.registerCallback(self.subscribers_callback)

        # Print initial state of the agent
        result_str = (
            "Active inference [ average force = " + str(self.avg_force )   +
            " self.self.f_ext_gradient_max = " + str(self.f_ext_gradient_max) + "]"
            )

        rospy.loginfo(result_str)

        for i in range(0,self.num_modalities):
        
            result_str = (
                "Observation modality " + self.modality_names[i] + " = [" 
                + self.obs_idx_names[i][self.observation[i]] + "]"
                )
            rospy.loginfo(result_str)

        for i in range(0,self.num_factors):
            
            result_str = (
                "Belief inference: " + self.factor_names[i] + " = [" + self.states_idx_names[i][self.state[i]] + "]"
                )
            rospy.loginfo(result_str)
        
        # Create a publisher for publishing Image messages on the 'webcam_video' topic
        self.pub_video = rospy.Publisher(self.pub_video_topic_name, Image, queue_size=10)
        
        # Create a CvBridge to convert OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        # Open the webcam
        self.cap = cv2.VideoCapture(0)
        
    def subscriber_callbak_ee_pose_cmd(self, ee_pose_cmd: PoseStamped) -> None:
        self.ee_pose_cmd_x = ee_pose_cmd.pose.position.x


    def subscribers_callback(self, wrench_stamped, ee_pose) -> None:
        
        if self.counter_init < self.T_init_loop_length:
            self.counter_init = self.counter_init + 1
            self.compute_desired_stiffness(self.belief_state)
            self.publish_results()
            self.set_point.pose.position.x = ee_pose.position.x
            return

        # Update data WARNING: HARD CODED MINUS SIGN
        # Proper way to deal with this is to have the force in the same frame as the end effector pose and goal position
        self.f_ext_win.append(-wrench_stamped.wrench.force.x)
        self.ee_pose_win.append(ee_pose.position.x)


        # Update data
        # To avoid to long arrays, delete first num_samples_emo_window when the arrays arrives at num_samples_emo_window*2 elements
        if (
            len(self.f_ext_win) > self.T_inference_win * 1.1
            and len(self.ee_pose_win) > self.T_inference_win * 1.1 
            and len(self.f_ext_gradient_win) > self.T_inference_win * 1.1
        ):
            self.f_ext_win = self.f_ext_win[self.T_inference_win :]
            self.ee_pose_win = self.ee_pose_win[self.T_inference_win :]
            self.f_ext_gradient_win = self.f_ext_gradient_win[self.T_inference_win :]

        self.avg_force = np.average(self.f_ext_win) - self.initial_force_bias
        self.ee_pos_x = self.ee_pose_win[-1]

        self.f_ext_gradient_win = np.gradient(self.f_ext_win)

        self.f_ext_gradient_max = np.max(self.f_ext_gradient_win)

        # Use force and ee pose to determine the observation
        if abs(self.avg_force) > self.high_force_threshold and np.sign(self.avg_force) == np.sign(self.goal_positions[0] - self.ee_pos_x ):
            self.observation[0] = TOWARDS_GAOL_1_HIGH_FORCE_EVIDENCE
        elif abs(self.avg_force) > self.high_force_threshold and np.sign(self.avg_force) == np.sign( self.goal_positions[1] - self.ee_pos_x):
            self.observation[0] = TOWARDS_GAOL_2_HIGH_FORCE_EVIDENCE
        else:
            self.observation[0] = AROUND_ZERO_FORCE_EVIDENCE

        # No noise level in this observation
        if (abs(self.f_ext_gradient_max) > self.f_ext_gradient_thresh) and (self.observation[1] == GO_TO_G1_0 or self.observation[1] == GO_TO_G2_0):
            self.observation[0] = NOT_SMOOTH_FORCE_EVIDENCE
        elif (abs(self.f_ext_gradient_max) < self.f_ext_gradient_thresh) and (self.observation[1] == GO_TO_G1_0 or self.observation[1] == GO_TO_G2_0):
            self.observation[0] = SMOOTH_FORCE_EVIDENCE

        self.counter = self.counter + 1
        
        # If we are close to the set point and we have enough samples to compute the inference
        if(abs(self.ee_pose_cmd_x - self.set_point.pose.position.x) < CLOSE_TO_SETPOINT_DISTANCE and self.counter > self.T_inference_win):

            self.compute_results()

            self.counter = 0
        
            result_str = (
            "Active inference [ average force = " + str(self.f_ext_win )   +
            " self.self.f_ext_gradient_max = " + str(self.f_ext_gradient_max) + "]"
            )

            rospy.loginfo(result_str)

            for i in range(0,self.num_modalities):
            
                result_str = (
                    "Observation modality " + self.modality_names[i] + " = [" 
                    + self.obs_idx_names[i][self.observation[i]] + "]"
                    )
                rospy.loginfo(result_str)

            for i in range(0,self.num_factors):
                
                result_str = (
                    "Belief inference: " + self.factor_names[i] + " = [" + str(self.belief_state[i]) + "]"
                    )
                rospy.loginfo(result_str)

        self.publish_results()

    def compute_results(self):

        # update agent
        self.belief_state = self.agent.infer_states(self.observation)
        self.agent.infer_policies()
        action = self.agent.sample_action()
        
        # Map control states to set-points
        if(int(self.state[1])==0):
            x_d = self.ee_pos_x
        elif (int(self.state[1])==1):
            if abs(self.goal_positions[0] - self.ee_pos_x) > CLOSE_TO_GOAL_DISTANCE: 
                x_d = self.explorative_dev*((self.goal_positions[0] - self.ee_pos_x)/np.linalg.norm(self.goal_positions[0] - self.ee_pos_x)) + self.ee_pos_x
            else:
                x_d = self.goal_positions[0]
        elif (int(self.state[1])==2):
            if abs(self.goal_positions[1] - self.ee_pos_x) > CLOSE_TO_GOAL_DISTANCE: 
                 x_d = self.explorative_dev*((self.goal_positions[1] - self.ee_pos_x)/np.linalg.norm(self.goal_positions[1] - self.ee_pos_x)) + self.ee_pos_x
            else:
                x_d = self.goal_positions[1]
        elif (int(self.state[1])==3):
            x_d = self.goal_positions[0]
        elif (int(self.state[1])==4):
            x_d = self.goal_positions[1]
        else:
            x_d = self.ee_pos_x

        self.set_point.pose.position.x = x_d
        # update environment
        for f, s in enumerate(self.state):
            self.state[f] = utils.sample(self.B_gp[f][:, s, int(action[f])])

        # Update self observation only, force and right/wrong are updated by looking at external signals
        self.observation[1] = utils.sample(self.A_gp[1][:, self.state[0], self.state[1], self.state[2]])

        # Update the stiffness of the robot
        self.compute_desired_stiffness(self.belief_state)

    def compute_desired_stiffness(self, belief_state):
        '''
        Compute the desired stiffness based on the belief state
        '''
        self.stiffness_desired = self.K_0 + ((1 - belief_state[2][1]) * (self.K_max - self.K_min)) + self.K_min
    
    def publish_results(self):
        
        if self.log_more_info:
            result_str = (
                "Active inference setpoint = ["
                + str(self.set_point.pose.position.x) + ", "
                + str(self.set_point.pose.position.y) + ", "
                + str(self.set_point.pose.position.z) + "]"
                )

            rospy.loginfo(result_str)

        # Set the time stamp of the results
        self.set_point.header.stamp = rospy.Time.now()
        
        self.pub.publish(self.set_point)

        self.belief_msg.goal_state_values = self.belief_state[0].tolist()  # Assuming state_values is a list of ndarrays
        self.belief_msg.motion_state_values = self.belief_state[1].tolist()
        self.belief_msg.proactive_state_values = self.belief_state[2].tolist()
        # Populate other fields as necessary
        self.belief_publisher.publish(self.belief_msg)

        # Publish the stiffness
        self.pub_stiffness.publish(self.stiffness_desired)

        # New code for publishing webcam video
        ret, frame = self.cap.read()
        if ret:
            try:
                image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.pub_video.publish(image_message)
            except cv2.CvBridgeError as e:
                rospy.logerr(e)

if __name__ == "__main__":
    try:
        # Initialize ros node
        rospy.init_node("active_inference_planner_node", anonymous=False)
        # Initialize class
        aip = ActiveInferencePlanner()
        # Block until ROS is shutdown
        rospy.spin()
        aip.cap.release()



    except rospy.ROSInterruptException:
        # Release the webcam
        aip.cap.release()
        pass
