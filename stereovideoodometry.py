import numpy as np
import cv2
import os

class StereoVideoOdometery(object):
    def __init__(self,
                 left_img_path,
                 right_img_path,
                 pose_file_path, # This parameter is not used in your current class logic
                 focal_length=718.8560,
                 pp=(607.1928, 185.2157),
                 lk_params=dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
                 detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
                 baseline=0.54):
        
        self.left_file_path = left_img_path
        self.right_file_path = right_img_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.baseline = baseline

        self.P_left = np.array([[focal_length, 0, pp[0], 0],
                                [0, focal_length, pp[1], 0],
                                [0, 0, 1, 0]])
        self.P_right = np.array([[focal_length, 0, pp[0], -focal_length * baseline],
                                 [0, focal_length, pp[1], 0],
                                 [0, 0, 1, 0]])
        
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        
        self.id = 0
        self.n_features = 0
        self.points_3d_local = None

        # Store keypoints and matches for visualization
        self.current_display_frame = None
        self.current_kpts_left = None
        self.current_kpts_right = None
        self.current_matches_mask = None # For successful optical flow matches

        self.process_first_frame()

    def hasNextFrame(self):
        num_frames = len(os.listdir(self.left_file_path))
        return self.id < num_frames

    def detect(self, img):
        p0 = self.detector.detect(img)
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def triangulate_points(self, p1, p2):
        if p1 is None or p2 is None or p1.shape[0] == 0 or p2.shape[0] == 0:
            return np.array([])
        p1_reshaped = p1.reshape(-1, 2).T
        p2_reshaped = p2.reshape(-1, 2).T
        points_4d_hom = cv2.triangulatePoints(self.P_left, self.P_right, p1_reshaped, p2_reshaped)
        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        return points_3d.T

    def process_first_frame(self):
        left_img_path = os.path.join(self.left_file_path, str(self.id).zfill(6) + '.png')
        self.old_frame_left = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        self.old_frame_right = cv2.imread(os.path.join(self.right_file_path, str(self.id).zfill(6) + '.png'), cv2.IMREAD_GRAYSCALE)

        self.p0_left = self.detect(self.old_frame_left)
        p0_right, st, _ = cv2.calcOpticalFlowPyrLK(self.old_frame_left, self.old_frame_right, self.p0_left, None, **self.lk_params)
        
        good_mask = st.flatten() == 1
        self.p0_left = self.p0_left[good_mask]
        good_p0_right = p0_right[good_mask]
        
        self.points_3d_local = self.triangulate_points(self.p0_left, good_p0_right)
        self.id += 1

        # Store for visualization in the first frame
        self.current_kpts_left = self.p0_left
        self.current_kpts_right = good_p0_right
        self.current_matches_mask = good_mask.astype(bool) # Ensure boolean mask
        self.current_display_frame = self.old_frame_left.copy() # Initialize with the first frame

    def process_frame(self):
        left_img_path = os.path.join(self.left_file_path, str(self.id).zfill(6) + '.png')
        current_frame_left = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        current_frame_right = cv2.imread(os.path.join(self.right_file_path, str(self.id).zfill(6) + '.png'), cv2.IMREAD_GRAYSCALE)
        
        # Reset visualization data for the current frame
        self.current_kpts_left = None
        self.current_kpts_right = None
        self.current_matches_mask = None

        if self.p0_left.shape[0] < 2000:
            # Re-detect features on the old left frame
            self.p0_left = self.detect(self.old_frame_left)
            p0_right_re, st_re, _ = cv2.calcOpticalFlowPyrLK(self.old_frame_left, self.old_frame_right, self.p0_left, None, **self.lk_params)
            
            good_mask_re = st_re.flatten() == 1
            self.p0_left = self.p0_left[good_mask_re]
            good_p0_right_re = p0_right_re[good_mask_re]
            
            self.points_3d_local = self.triangulate_points(self.p0_left, good_p0_right_re)

            # Store for visualization if re-detection happened
            self.current_kpts_left = self.p0_left
            self.current_kpts_right = good_p0_right_re
            self.current_matches_mask = good_mask_re.astype(bool)

        p1_left, st, _ = cv2.calcOpticalFlowPyrLK(self.old_frame_left, current_frame_left, self.p0_left, None, **self.lk_params)
        good_mask = st.flatten() == 1
        
        points_3d_for_pnp = self.points_3d_local[good_mask]
        good_p1_left = p1_left[good_mask]
        
        K = self.P_left[:, :3]
        if points_3d_for_pnp.shape[0] < 5:
            self.id += 1
            # If not enough points, just update the frame and continue
            self.old_frame_left = current_frame_left
            self.old_frame_right = current_frame_right
            self.current_display_frame = current_frame_left.copy()
            return

        _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d_for_pnp, good_p1_left, K, distCoeffs=None)
        R_pnp, _ = cv2.Rodrigues(rvec)

        R_motion = R_pnp.T
        t_motion = -R_motion @ tvec
        
        self.t = self.t + self.R @ t_motion
        self.R = self.R @ R_motion
        
        p1_right, st_stereo, _ = cv2.calcOpticalFlowPyrLK(current_frame_left, current_frame_right, good_p1_left, None, **self.lk_params)
        good_stereo_mask = st_stereo.flatten() == 1
        good_p1_left_stereo = good_p1_left[good_stereo_mask]
        good_p1_right_stereo = p1_right[good_stereo_mask]

        self.points_3d_local = self.triangulate_points(good_p1_left_stereo, good_p1_right_stereo)
        self.p0_left = good_p1_left_stereo.reshape(-1, 1, 2)
        
        self.old_frame_left = current_frame_left
        self.old_frame_right = current_frame_right
        
        self.id += 1

        # Store for visualization in the regular processing frame
        self.current_kpts_left = good_p1_left_stereo
        self.current_kpts_right = good_p1_right_stereo
        self.current_matches_mask = good_stereo_mask.astype(bool)

        # Prepare the frame for display with keypoints and matches
        self.current_display_frame = self._draw_features_on_frame(current_frame_left, current_frame_right)
    
    def _draw_features_on_frame(self, left_img, right_img):
        # Convert to color image for drawing
        display_frame = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)

        if self.current_kpts_left is not None and self.current_kpts_right is not None:
            for i in range(self.current_kpts_left.shape[0]):
                pt_left = (int(self.current_kpts_left[i, 0, 0]), int(self.current_kpts_left[i, 0, 1]))
                # Draw keypoint on left image (green)
                cv2.circle(display_frame, pt_left, 3, (0, 255, 0), -1)

                if self.current_matches_mask[i]:
                    pt_right = (int(self.current_kpts_right[i, 0, 0]), int(self.current_kpts_right[i, 0, 1]))
                    # Draw a line from the left keypoint to its corresponding right keypoint
                    # (offsetting the right keypoint's X by the width of the left image)
                    # For simplicity, we are drawing features only on the left image
                    # You could create a concatenated image of left and right to draw lines between them.
                    # For now, let's just highlight the points on the left image that have matches.
                    pass # We only draw points on the left image in this current setup.

        return display_frame
    
    def get_world_pose(self):
        return self.R, self.t
    
    def get_local_points(self):
        return self.points_3d_local
        
    def get_current_frame_for_display(self):
        # Return the processed frame with features drawn
        if self.current_display_frame is None:
            # Fallback if no frame has been processed yet, or no features to draw
            return self.old_frame_left.copy() 
        return self.current_display_frame