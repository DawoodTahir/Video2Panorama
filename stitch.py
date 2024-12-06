import cv2
import numpy as np
import os
import time
from collections import defaultdict

# Timer class to track execution time of various operations
class Timer:
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if name in self.start_times:
            elapsed_time = time.perf_counter() - self.start_times[name]
            self.timings[name].append(elapsed_time)
            del self.start_times[name]

    def get_statistics(self):
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': sum(times),
                'calls': len(times)
            }
        return stats

    def print_statistics(self):
        stats = self.get_statistics()
        print("\nPerformance Statistics:")
        print("-" * 80)
        print(f"{'Operation':<25} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'Total (s)':>10} {'Calls':>8}")
        print("-" * 80)
        for name, stat in stats.items():
            print(f"{name:<25} {stat['mean']*1000:10.2f} {stat['std']*1000:10.2f} {stat['min']*1000:10.2f} "
                  f"{stat['max']*1000:10.2f} {stat['total']:10.2f} {stat['calls']:8d}")

# Function to project an image onto a cylindrical surface
def cylindrical_projection(img, focal_length, correction_factor=0.6):
    h, w = img.shape[:2]
    K = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length * correction_factor, h / 2],
        [0, 0, 1]
    ])
    cyl = np.zeros_like(img)
    
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    theta = (x_coords - w / 2) / focal_length
    h_ = (y_coords - h / 2) / (focal_length * correction_factor)
    ##claculting of cylindrical frame projections 
    X = np.sin(theta)
    Y = h_
    Z = np.cos(theta)
    ##converting cylindrical projection back to 2D to put onto blank image
    x_ = focal_length * (X / Z) + w / 2
    y_ = focal_length * correction_factor * (Y / Z) + h / 2
    
    # Create a mask for valid coordinates to avoid out-of-bound errors
    valid = (x_ >= 0) & (x_ < w) & (y_ >= 0) & (y_ < h)
    x_valid = x_[valid].astype(np.int32)
    y_valid = y_[valid].astype(np.int32)
    cyl[y_coords[valid], x_coords[valid]] = img[y_valid, x_valid]
    
    return cyl

# VideoMosaic class handles the creation of the panorama
class VideoMosaic:
    def __init__(self, first_frame, focal_length=500, detector_type="sift", correction_factor=0.6):
        self.timer = Timer()
        self.detector_type = detector_type
        
        # Initialize feature detector and FLANN-based matcher
        self.timer.start('init_detector')
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(nfeatures=2000)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(nfeatures=2000)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
        else:
            raise ValueError("Unsupported detector type.")
                
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)  # FLANN-based matcher
        self.timer.stop('init_detector')
        
        self.focal_length = focal_length
        self.correction_factor = correction_factor
        
        # Project the first frame cylindrically
        self.timer.start('first_frame_projection')
        self.first_frame_cyl = cylindrical_projection(first_frame, focal_length, correction_factor)
        self.timer.stop('first_frame_projection')
        
        h, w = self.first_frame_cyl.shape[:2]
        self.panorama = self.first_frame_cyl.copy()  # Initialize panorama
        self.transforms = [np.eye(3, dtype=np.float32)]  # List to hold cumulative transformations
        
        self.prev_frame = self.first_frame_cyl
        gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors for the first frame
        self.timer.start('initial_feature_detection')
        self.prev_keypoints, self.prev_descriptors = self.detector.detectAndCompute(gray, None)
        self.timer.stop('initial_feature_detection')
        
        # if self.detector_type == "orb":
        #     self.prev_descriptors = np.float32(self.prev_descriptors)  # Convert ORB descriptors to float32
        
        # # Set up visualization windows
        # cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Feature Matches", 800, 400)
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Panorama", 800, 400)

    # Method to process each frame and update the panorama
    def process_frame(self, frame, frame_idx, show_matches=False):
        # Cylindrical projection of the current frame
        # self.timer.start('frame_projection')
        frame_cyl = cylindrical_projection(frame, self.focal_length, self.correction_factor)
        # self.timer.stop('frame_projection')
        
        # Preprocessing: convert to grayscale
        # self.timer.start('preprocessing')
        gray = cv2.cvtColor(frame_cyl, cv2.COLOR_BGR2GRAY)
        # self.timer.stop('preprocessing')
        
        # Feature detection and description
        # self.timer.start('feature_detection')
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        # self.timer.stop('feature_detection')
        
        # Skip frame if no descriptors are found
        if descriptors is None or self.prev_descriptors is None:
            print(f"Skipping frame {frame_idx}: No descriptors found.")
            return
            
        # if self.detector_type == "orb":
        #     descriptors = np.float32(descriptors)  # Convert ORB descriptors to float32
        
        # Feature matching using FLANN-based matcher
        # self.timer.start('feature_matching')
        matches = self.matcher.knnMatch(descriptors, self.prev_descriptors, k=2)
        good_matches = []
        
        # Apply Lowe's ratio test to filter good matches
        for i in range(len(matches)):
            if len(matches[i]) == 2:
                m, n = matches[i]

                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        # self.timer.stop('feature_matching')
        
        # Skip frame if not enough good matches
        if len(good_matches) < 10:
            print(f"Skipping frame {frame_idx}: Not enough good matches.")
            return
        
        # Visualization of feature matches (optional)
        # if show_matches:
        #     self.timer.start('visualization')
        #     match_img = cv2.drawMatches(
        #         frame_cyl, keypoints,
        #         self.prev_frame, self.prev_keypoints,
        #         good_matches, None, flags=2
        #     )
        #     cv2.imshow("Feature Matches", match_img)
        #     cv2.waitKey(1)
            self.timer.stop('visualization')
        
        # Extract matched keypoints' coordinates
        self.timer.start('transform_estimation')
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.prev_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate affine transformation using RANSAC
        H, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        self.timer.stop('transform_estimation')
        
        # Skip frame if transformation couldn't be computed
        if H is None:
            print(f"Skipping frame {frame_idx}: Transform could not be computed.")
            return
        
        # Adjust vertical translation to reduce misalignment
        H[1, 2] *= 0.2
        H = np.vstack([H, [0, 0, 1]]).astype(np.float32)  # Convert to homography matrix
        self.transforms.append(self.transforms[-1] @ H)  # Accumulate transformations
        
        # Panorama update: warp current frame and blend with existing panorama
        self.timer.start('panorama_update')
        h_pano, w_pano = self.panorama.shape[:2]
        h_frame, w_frame = frame_cyl.shape[:2]
        
        # Define corners of the current frame
        corners = np.array([
            [0, 0, 1],
            [w_frame, 0, 1],
            [w_frame, h_frame, 1],
            [0, h_frame, 1]
        ])
        
        # Transform corners using the accumulated homography
        transformed_corners = (self.transforms[-1] @ corners.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
        
        # Determine new panorama boundaries
        min_x = min(0, transformed_corners[:, 0].min())
        max_x = max(w_pano, transformed_corners[:, 0].max())
        min_y = min(0, transformed_corners[:, 1].min())
        max_y = max(h_pano, transformed_corners[:, 1].max())
        
        # Create a translation matrix to shift panorama if needed
        T = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Calculate new panorama size
        new_width = int(max_x - min_x)
        new_height = int(max_y - min_y)
        
        # Warp the current frame into the panorama space
        warped_frame = cv2.warpPerspective(frame_cyl, T @ self.transforms[-1], (new_width, new_height))
        # Warp the existing panorama to the new space
        warped_pano = cv2.warpPerspective(self.panorama, T, (new_width, new_height))
        
        # Create a mask for blending: regions where warped frame has pixels
        mask = (warped_frame > 0).astype(np.uint8)
        # Blend the warped frame and existing panorama
        self.panorama = cv2.add(warped_pano * (1 - mask), warped_frame)
        self.timer.stop('panorama_update')
        
        # Update previous frame information for next iteration
        self.prev_frame = frame_cyl
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        if self.detector_type == "orb":
            self.prev_descriptors = np.float32(self.prev_descriptors)
        
        # Display the updated panorama
        cv2.imshow("Panorama", self.panorama)
        cv2.waitKey(1)

    # Method to save the final panorama and print performance statistics
    def save_output(self, output_path):
        cv2.namedWindow("Final Panorama", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Final Panorama", 800, 400)
        
        cv2.imwrite(output_path, self.panorama)  # Save panorama to disk
        print(f"Mosaic saved at {output_path}")
        
        # Print timing statistics
        self.timer.print_statistics()
        
        cv2.imshow("Final Panorama", self.panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main function to execute the panorama stitching process
def main(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return
    
    total_timer = Timer()
    total_timer.start('total_execution')  # Start total execution timer
    
    cap = cv2.VideoCapture(video_path)  # Open video file
    if not cap.isOpened():
        print("Could not open video file.")
        return
    
    ret, first_frame = cap.read()  # Read the first frame
    if not ret:
        print("Error reading the first frame.")
        return
    
    # Resize the first frame for faster processing (50% of original size)
    scale_percent = 50
    width = int(first_frame.shape[1] * scale_percent / 100)
    height = int(first_frame.shape[0] * scale_percent / 100)
    first_frame = cv2.resize(first_frame, (width, height), interpolation=cv2.INTER_AREA)
    
    # Initialize the VideoMosaic object with FLANN-based matcher
    mosaic = VideoMosaic(
        first_frame,
        focal_length=width * 1.25,  # Adjust focal length based on frame width
        detector_type="sift",        # Use SIFT detector
        correction_factor=0.6        # Set vertical scaling correction
    )
    
    frame_skip = 5  # Process every 5th frame for efficiency
    frame_idx = 0   # Initialize frame index
    frames_processed = 0  # Counter for processed frames
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames left
        
        # Resize current frame to match the first frame's size
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Process every nth frame based on frame_skip
        if frame_idx % frame_skip == 0:
            mosaic.process_frame(frame, frame_idx, show_matches=False)  # Show feature matches
            frames_processed += 1
        
        frame_idx += 1  # Increment frame index
        
        # Allow user to quit early by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()  # Release video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
    
    total_timer.stop('total_execution')  # Stop total execution timer
    total_execution_time = total_timer.get_statistics()['total_execution']['total']
    fps = frames_processed / total_execution_time  # Calculate average FPS
    print(f"\nProcessing Statistics:")
    print(f"Total frames processed: {frames_processed}")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    mosaic.save_output(output_path)  # Save the final panorama

# Entry point of the script
if __name__ == "__main__":
    video_path = "data_3.MP4"  # Path to input video
    output_path = "mosaic_output_flan.jpg"  # Path to save the panorama
    main(video_path, output_path)
