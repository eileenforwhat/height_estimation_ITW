import cv2
import numpy as np

def get_depth_map(im1_path, im2_path):
    # Load left and right stereo images
    left_image = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    # Stereo block matching parameters (adjust these as needed)
    min_disparity = 0
    max_disparity = 32
    block_size = 5

    # Compute disparity map using stereo block matching
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=max_disparity - min_disparity,
        blockSize=block_size,
    )
    disparity_map = stereo.compute(left_image, right_image)

    # Normalize the disparity map to 0-255 and convert to 8-bit format
    normalized_disparity = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_8bit = np.uint8(normalized_disparity)

    # Calculate depth map from disparity map and camera calibration parameters
    # Replace these values with your actual camera calibration data
    baseline = 31.27970083866741  # Baseline distance between the two cameras (in meters)
    focal_length = 1000.0  # Focal length (in pixels)

    depth_map = (baseline * focal_length) / disparity_map

    # Display and save the depth map
    cv2.imshow('Depth Map', depth_map)
    cv2.imwrite('depth_map.png', depth_map)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    im1_path, im2_path = ('data/cathedral/IMG_6343.jpeg', 'data/cathedral/IMG_6346.jpeg')
    get_depth_map(im1_path, im2_path)
