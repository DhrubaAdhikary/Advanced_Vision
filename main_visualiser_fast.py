import numpy as np
import cv2 as cv
from stereovideoodometry import StereoVideoOdometery
from render import Renderer3D 
import argparse
import pygame # Make sure to import pygame here as well for its quit() function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_path', type=str, default='dataset/sequences/00/image_0')
    parser.add_argument('--right_path', type=str, default='dataset/sequences/00/image_1')
    parser.add_argument('--pose_path', type=str, default='dataset/poses/00.txt') # This argument is not used in the provided StereoVideoOdometery class
    args = parser.parse_args()

    vo = StereoVideoOdometery(args.left_path, args.right_path, args.pose_path)
    renderer = Renderer3D(cam_distance=200)
    cv.namedWindow('Camera View', cv.WINDOW_NORMAL)

    all_poses = []
    # Change: Use a list to store all points for persistent map
    point_map_list = [] 

    while vo.hasNextFrame():
        vo.process_frame()
        
        R_world, t_world = vo.get_world_pose()
        points_local = vo.get_local_points()
        
        if points_local is not None and points_local.shape[0] > 0:
            points_world = (R_world @ points_local.T).T + t_world.flatten()

            # Ensure we don't try to sample more points than available
            num_to_sample = min(len(points_world), 150) 
            if num_to_sample > 0:
                sample_indices = np.random.choice(points_world.shape[0], num_to_sample, replace=False)

                for point in points_world[sample_indices]:
                    gl_point = point.copy()
                    gl_point[1] *= -1 
                    gl_point[2] *= -1
                    point_map_list.append(gl_point) # Add to the list
        
        # =======================================================================
        # --- THE FINAL FIX ---
        # Store Pose for Trajectory Rendering, ensuring Z is also flipped for OpenGL
        # =======================================================================
        gl_t = t_world.copy().flatten()
        gl_t[1] *= -1 # Invert Y for OpenGL
        gl_t[2] *= -1 # Invert Z for OpenGL to match the points
        # =======================================================================

        all_poses.append({'t': gl_t, 'R': R_world})

        # Convert the list of points to a NumPy array for rendering
        points_to_render = np.array(point_map_list) if point_map_list else np.array([])
        renderer.render3dSpace(points_to_render, all_poses)

        # Get the current frame with overlayed features for display
        display_frame = vo.get_current_frame_for_display()
        cv.imshow('Camera View', display_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    pygame.quit() # Ensure pygame is properly quit

if __name__ == "__main__":
    main()