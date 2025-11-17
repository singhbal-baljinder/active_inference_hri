import cv2
import numpy as np
import glob
import time
import yaml
import os
import matplotlib.pyplot as plt
from active_inference_planner.utils import *
import argparse
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from numpy.linalg import inv
import tf
from scipy.spatial.transform import Rotation as R

class CameraCalibrator:
    # Intrinsics and distortion parameters of the calibrated camera model
    camera_matrix = None
    dist_coeffs = None

    pose_objs = []

    # Dimensions of A4 paper in pixels at 300 DPI
    a4_width = 600
    a4_height = 500
    # Load the Aruco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()

    min_detected_markers = 4
    B_T_C = None
    C_T_P = None
    B_T_P = None
    use_camera_based_calib = False
    calibration_data = {}

    def __init__(self, squares_x, squares_y, square_length, marker_length, margin_size=50,  dictionary=cv2.aruco.DICT_6X6_250):
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.dictionary = dictionary
        
        self.charuco_board = cv2.aruco.CharucoBoard_create(
            squaresX=self.squares_x, squaresY=self.squares_y, squareLength=self.square_length, markerLength=self.marker_length,
            dictionary=cv2.aruco.Dictionary_get(dictionary)
        )
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.all_corners = []
        self.all_ids = []
        self.image_size = None
        self.margin_size = margin_size

        self.bridge = CvBridge()

    def print_board(self, output_folder='.', output_file='charuco_board.jpg'):
        # Create the output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Create the Charuco board image
        self.width = self.squares_x * self.square_length + 2 * self.margin_size
        self.height = self.squares_y * self.square_length + 2 * self.margin_size
        board_image = self.charuco_board.draw((self.width, self.height), marginSize=self.margin_size)

        # Save the Charuco board image
        output_path = os.path.join(output_folder, output_file)
        cv2.imwrite(output_path, board_image)
        print(f"Charuco board saved as {output_path}")


    def save_images_from_video(self, video_source=0, delay=1, output_folder='.', max_frames=20):
        # Create the output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_source)
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.charuco_board.dictionary)
            
            if ids is None:
                continue
            if len(ids) < self.min_detected_markers:
               continue

            print(f"Detected {len(ids)} markers")

            frame_count += 1
            output_path = f"{output_folder}/frame_{frame_count}.jpg"
            cv2.imwrite(output_path, frame)
            print(f"Captured {output_path}")

            time.sleep(delay)

        cap.release()

    def save_images_from_ros(self, video_source_topic='camera/image_raw', output_folder='.', max_frames=10):
        
        rospy.init_node('camera_calibrator')
        self.sub_camera_image = rospy.Subscriber(video_source_topic, Image, self.save_image_callback)
        self.bridge = CvBridge()
        self.cv_image = None
        self.frame_count = 0
        self.output_folder = output_folder
        self.max_frames = max_frames
        self.counter_sub_time = 0
        self.sub_freq = 100

        rospy.spin()

        
    def save_image_callback(self, camera_image: Image):
        '''
        Get images from ROS topic and save them to disk.
        '''
        if camera_image is None:
            return
        self.counter_sub_time += 1
        if self.counter_sub_time % self.sub_freq != 0:
            return
        if self.counter_sub_time // self.sub_freq == self.max_frames:
            rospy.signal_shutdown("Max frames reached.")
            return
        
        print("Received image from ROS topic.")
        self.cv_image = self.bridge.imgmsg_to_cv2(camera_image, "bgr8")

        # Save the image only if it has enough markers
        corners, ids, _ = cv2.aruco.detectMarkers(self.cv_image, self.charuco_board.dictionary)
        
        if ids is None or len(ids) < self.min_detected_markers:
            return
        else:
            output_path = f"{self.output_folder}/frame_{self.frame_count}.jpg"
            cv2.imwrite(output_path, self.cv_image)
            print(f"Captured {output_path}")
            self.frame_count += 1
        if self.frame_count >= self.max_frames:
            rospy.signal_shutdown("Max frames reached.")
    
    def calibrate_camera(self, folder_path: str):
        self.camera_matrix = np.zeros((3, 3))
        self.dist_coeffs = np.zeros((5, 1))

        images = glob.glob(f'{folder_path}/*.jpg')
        for image_file in images:
            image = cv2.imread(image_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.image_size = gray.shape[::-1]

            # Detect Aruco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.charuco_board.dictionary)

            if ids is not None and len(ids) >= self.min_detected_markers:
                # Refine detected markers
                cv2.aruco.refineDetectedMarkers(gray, self.charuco_board, corners, ids, rejectedCorners=None)

                # Interpolate Charuco corners
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
                

                if retval > 0 and len(charuco_corners) > self.min_detected_markers and len(charuco_ids) > 0:
                    self.all_corners.append(charuco_corners)
                    self.all_ids.append(charuco_ids)
                else:
                    print(f"Charuco corners interpolation failed in {image_file}.")
            else:
                print(f"No valid Charuco corners or IDs found in {image_file}.")

        if len(self.all_corners) == 0 or len(self.all_ids) == 0:
            print("No valid Charuco corners or IDs found. Calibration failed.")
            return

        retval, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self.all_corners,
            charucoIds=self.all_ids,
            board=self.charuco_board,
            imageSize=self.image_size,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs
        )

        # Compute reprojection error
        mean_error = 0
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        for i in range(len(self.all_corners)):

            imgpoints, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs)


        # Save the calibration results to a YAML file
        calibration_data = {
            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'camera_matrix': self.camera_matrix.flatten().tolist(),
            'distortion_coefficients': self.dist_coeffs.flatten().tolist()
        }

        with open(os.path.join(folder_path, 'camera_calibration.yaml'), 'w') as file:
            yaml.dump(calibration_data, file, default_flow_style=True)

        print("Calibration successful!")
        print("Camera matrix:\n", self.camera_matrix.flatten().tolist())
        print("Distortion coefficients:\n", self.dist_coeffs.flatten().tolist())
    
    def compute_C_T_P(self, image: np.ndarray, calibration_data: dict) -> dict:
        '''
        Given a camera model, compute the pose of a pattern wrt the camera.
        '''

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, self.charuco_board.dictionary)

        # Load the camera matrix and distortion coefficients
        if 'camera_matrix' in calibration_data and 'distortion_coefficients' in calibration_data:
            camera_matrix = np.array(calibration_data['camera_matrix']).reshape((3, 3))
            dist_coeffs = np.array(calibration_data['distortion_coefficients'])
            print("[compute_C_T_P]: Camera matrix and distortion coefficients loaded.")
            print("[compute_C_T_P]: Camera matrix:\n", camera_matrix)
            print("[compute_C_T_P]: Distortion coefficients:\n", dist_coeffs)
        else:
            print("[compute_C_T_P]: Camera matrix and distortion coefficients not found in calibration data.")
            return None

        if ids is not None:
            # Interpolate Charuco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray_image, self.charuco_board)

            if retval > 0:
                # Estimate the pose of the Charuco board
                rvec = np.zeros((3, 1))
                tvec = np.zeros((3, 1))
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, camera_matrix, dist_coeffs, rvec, tvec)
                
                if retval:
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)

                    # Create the 4x4 transformation matrix
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = tvec.flatten()

                    # Save the transformation matrix to a YAML file
                    pose_data = {
                        'C_T_P': transformation_matrix.flatten().tolist()
                    }

                    print("[compute_C_T_P]: Pose estimation successful. Transformation matrix C_T_P:\n", transformation_matrix)
                
                    
                    #Print detected markers
                    img = cv2.aruco.drawDetectedMarkers(image, corners, ids)

                    cv2.imshow('Pose', img)
                    cv2.waitKey(0)

                    # Print image with pose
                    axis_length = 0.1
                    img = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

                    cv2.imshow('Pose', img)
                    cv2.waitKey(0)

                    cv2.destroyAllWindows()

                    return pose_data

                else:
                    print("[compute_C_T_P]: pose estimation failed.")
            else:
                print("[compute_C_T_P]: Charuco corners interpolation failed.")
        else:
            print("[compute_C_T_P]: no markers detected.")

        return None

    def compute_B_T_C(self, filename=None, calibration_data=None) -> dict:
        '''
        Compute frame transformotion from camera to robot base.
        Frame legend:
            B -> Base
            C -> Camera
            EE - > End-Effector
            H -> Hand
            P -> Pattern
        '''
        # Get B_T_EE (computed using FK), EE_T_H (end-effector <-> hand tip), H_T_P (hand tip <-> Pattern), C_T_P (Camera <-> Pattern)
        if calibration_data is None:
            if filename is None:
                print("[compute_B_T_C]: Please provide a calibration file.")
                return
            with open(filename, 'r') as file:
                calibration_data = yaml.safe_load(file)
                if 'B_T_EE' in calibration_data and 'EE_T_H' in calibration_data and 'H_T_P' in calibration_data and 'C_T_P' in calibration_data:
                    B_T_EE = np.array(calibration_data['B_T_EE']).reshape((4, 4))
                    EE_T_H = np.array(calibration_data['EE_T_H']).reshape((4, 4))
                    H_T_P = np.array(calibration_data['H_T_P']).reshape((4, 4))
                    C_T_P = np.array(calibration_data['C_T_P']).reshape((4, 4))
                else:
                    print("[compute_B_T_C]: Required transforms not found in calibration data.")
                    return
        else:
            if 'B_T_EE' in calibration_data and 'EE_T_H' in calibration_data and 'H_T_P' in calibration_data and 'C_T_P' in calibration_data:
                B_T_EE = np.array(calibration_data['B_T_EE']).reshape((4, 4))
                EE_T_H = np.array(calibration_data['EE_T_H']).reshape((4, 4))
                H_T_P = np.array(calibration_data['H_T_P']).reshape((4, 4))
                C_T_P = np.array(calibration_data['C_T_P']).reshape((4, 4))
            else:
                print("[compute_B_T_C]: Required transforms not found in calibration data.")
                return

        # Compute B_T_C
        if C_T_P is None or H_T_P is None or EE_T_H is None or B_T_EE is None:
            print("[compute_B_T_C]: Cannot compute B_T_C without all the required transforms.")
            return
        
        B_T_C = B_T_EE @ EE_T_H @ H_T_P @ inv(C_T_P)

        print("[compute_B_T_C]: B_T_C computed successfully.")
        print("[compute_B_T_C]: B_T_C:\n", B_T_C.flatten().tolist())

        calibration_data.update({'B_T_C': B_T_C.flatten().tolist()})

        # Compute also the location of the pattern wrt the base (usefull when pattern is on the table used for experiments)
        B_T_P = B_T_C @ C_T_P
        calibration_data.update({'B_T_P': B_T_P.flatten().tolist()})

        return calibration_data

    def compute_B_T_C_using_ros(self, calibration_filepath='./calibration_file.yaml', video_source_topic='camera/color/image_raw', robot_pose_topic='/franka_state_controller/franka_states'):
        '''
        Compute B_T_C (camera to robot base transformation matrix)  using data from ROS topics.
        '''
        rospy.init_node('camera_calibrator')
        self.calibration_data = yaml.safe_load(open(calibration_filepath, 'r'))
        self.sub_robot_pose = message_filters.Subscriber(robot_pose_topic, FrankaState)
        self.sub_camera_image = message_filters.Subscriber(video_source_topic, Image)
        # Synchronize the subscribers
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_robot_pose, self.sub_camera_image],
            10,
            0.01,
            allow_headerless=True,
        )
        ts.registerCallback(self.subscribers_calibration_callback)
        rospy.spin()

    def subscribers_calibration_callback(self, robot_pose, camera_image):
        '''
        Callback function to receive the data from the robot and the camera in a synchronized fashion. 
        Then uses the following to compute the transformation matrix from the camera to the robot base:
         - the camera model;
         - image of the pattern located at  a known pose wrt the robot end-effector (pattern in hand configuration);
         - the pose of the end-effector wrt the robot base while taking the image.
        '''
        # Get force and ee pose
        # Franka stores the matrix in column major order while the yaml used in this script follows row major
        # So after reading the matrix we transpose it
        self.B_T_EE = np.array([robot_pose.O_T_EE]).reshape((4, 4)).T
        self.cv_image = self.bridge.imgmsg_to_cv2(camera_image, "bgr8")

        self.calibration_data.update({'B_T_EE': self.B_T_EE.flatten().tolist()})

        C_T_P_dict = self.compute_C_T_P(self.cv_image, self.calibration_data)

        if C_T_P_dict is None:
            return
        
        self.calibration_data.update(C_T_P_dict)

        cd = self.compute_B_T_C(calibration_data=self.calibration_data)

        if cd is None:
            return
        
        # Update calibration
        self.calibration_data.update(cd)
        # Compute validation error on pose translation part
        B_T_C_tmp = np.array(cd['B_T_C']).reshape((4, 4))
        C_T_P_tmp = np.array(cd['C_T_P']).reshape((4, 4))
        B_T_EE_tmp = np.array(cd['B_T_EE']).reshape((4, 4))
        # Assume EE and P are the same frames (in practice they differ by around 1cm dist along z, same orientation)
        B_T_EE_estimate = B_T_C_tmp @ C_T_P_tmp
        error_translation = np.linalg.norm(B_T_EE_estimate[0:3,3] - B_T_EE_tmp[0:3,3])
        
        print(f"[subscribers_calibration_callback]: B_T_C computed successfully. Training error on pose translation: {error_translation}")
        print("[subscribers_calibration_callback]: CALIBRATION DATA:\n", cd)
         
    def compute_pose_objs_ros(self, calibration_data: dict, pose_idx: int, image_source_topic='camera/color/image_raw', robot_pose_topic='/franka_state_controller/franka_states'):
        rospy.init_node('camera_calibrator')
        
        self.sub_robot_pose = message_filters.Subscriber(robot_pose_topic, FrankaState)
        self.sub_camera_image = message_filters.Subscriber(image_source_topic, Image)
        
        # Synchronize the subscribers
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_robot_pose, self.sub_camera_image],
            10,
            0.01,
            allow_headerless=True,
        )

        self.pose_idx = pose_idx
        self.calibration_data = calibration_data

        ts.registerCallback(self.calib_pose_objs_image_callback)

        rospy.spin()
    
    def calib_pose_objs_image_callback(self, robot_state: FrankaState, image: Image):

        self.cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        B_T_EE = np.array([robot_state.O_T_EE]).reshape((4, 4)).T

        if self.use_camera_based_calib:
            C_T_P_dict = self.compute_C_T_P(self.cv_image, self.calibration_data)

            if C_T_P_dict is None:
                return
            
            if 'C_T_P' in C_T_P_dict and  'B_T_C' in self.calibration_data:
                C_T_P1 = np.array(C_T_P_dict['C_T_P']).reshape((4, 4))
                B_T_C = np.array(self.calibration_data['B_T_C']).reshape((4, 4)) 
            else:
                rospy.logerr("B_T_C and C_T_P1 are required to compute B_T_P1")
                return

            B_T_P1 = B_T_C @ C_T_P1
        else:
            B_T_P1 = B_T_EE

        r = R.from_matrix(B_T_P1[:3, :3])
        q =  r.as_quat()


        px = B_T_P1[0, 3]
        py = B_T_P1[1, 3]
        pz = B_T_P1[2, 3]

        rotx = q[0]
        roty = q[1]
        rotz = q[2]
        rotw = q[3]
        
        self.calibration_data.update({'P'+str(self.pose_idx): [px,py,pz,rotx,roty,rotz, rotw]})

        # Compute validation error

        print(f"B_T_EE = \n {B_T_EE}")
        print(f"B_T_P1 = \n {B_T_P1}")
        error_translation = np.linalg.norm(B_T_P1[0:3,3] - B_T_EE[0:3,3])
        print(f"Pose object estimated with validation error: {error_translation}")

        print("calibration_data", self.calibration_data)

        return self.calibration_data

    def compute_pose_objects(self, cv_image, filename):

        # Detect the Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            # Estimate the pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.camera_matrix, self.dist_coeffs)
            self.pose_objs = []
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                pose = {
                    'id': int(ids[i]),
                    'rvec': rvec.tolist(),
                    'tvec': tvec.tolist(),
                    'corners': corners[i].tolist()
                }
                self.pose_objs.append(pose)


        with open(filename, 'w') as file:
            yaml.dump(self.pose_objs, file)
            
        print(f"Poses saved to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Calibration and Pose Estimation')
    parser.add_argument('--mode', type=str, required=True, choices=['print_board', 'calibrate_camera_model_camera', 'calibrate_camera_model_ros', 'estimate_C_T_P', 'estimate_B_T_C', 'estimate_pose_objects'], help='Mode of operation')
    parser.add_argument('--image_path', type=str, help='Path to the image for pose estimation or path to folder containing images for calibration of the camera model. \
                        It serves also as the output folder to store the results')
    parser.add_argument('--output_folder', type=str, help='Output folder for the captured images')
    parser.add_argument('--calibration_file', type=str, help='Calibration file for C_T_P and B_T_C estimation')
    parser.add_argument('--idx_pose', type=int, help='Index of the position to calibrate (int)')
    args = parser.parse_args()

    calibrator = CameraCalibrator(squares_x=5, squares_y=7, square_length=0.039, marker_length=0.023, margin_size=10)

    if args.mode == 'calibrate_camera_model_ros':
        
        if not args.image_path:
            print("Using images taken from video stream to calibrate the camera model.")
            calibrator.save_images_from_ros(video_source_topic='/camera/color/image_raw', output_folder=args.output_folder, max_frames=30)
            image_path = args.output_folder
        else:
            image_path = args.image_path

        calibrator.calibrate_camera(folder_path = image_path)
    
    elif args.mode == 'calibrate_camera_model_camera':
        print("Using images from video stream (no ROS) to calibrate the camera model.")
        calibrator.save_images_from_video(video_source=0, output_folder=args.output_folder, max_frames=30, delay=1)

        calibrator.calibrate_camera(folder_path = args.output_folder)
                                    

    elif args.mode == 'estimate_C_T_P':

        if args.image_path and args.calibration_file:
            calibration_data = yaml.safe_load(open(args.calibration_file, 'r'))

            image = cv2.imread(args.image_path)

            calibrator.compute_C_T_P(image, calibration_data)
        else:
            print("Please provide --image_path and calibration_file for C_T_P estimation.")

    elif args.mode == 'estimate_B_T_C':

        if args.calibration_file:
            calibrator.compute_B_T_C_using_ros(calibration_filepath=args.calibration_file)
        else:
            print("Please provide --calibration_file for B_T_C estimation.")

    elif args.mode == 'print_board':

        if args.image_path:
            calibrator.print_board(output_folder=args.image_path)
        else:
            calibrator.print_board()

    elif args.mode == 'estimate_pose_objects':

        if args.calibration_file:
            if args.idx_pose is not None:
                calibration_data = yaml.safe_load(open(args.calibration_file, 'r'))
                print(f"Idx pose = {args.idx_pose}")
                calibrator.compute_pose_objs_ros(calibration_data, pose_idx=args.idx_pose)  

