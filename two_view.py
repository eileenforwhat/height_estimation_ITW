# Two view reconstruction (w/o manual annots)
# 1. keypoint extraction and corresponding (SIFT) - done
# 2. Compute F - done
# 3. Compute P, P' from F - done
# 4. manually annotate peak - eileen
# 5. triangulate for the peak to get height - simon
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(images_dir, num_images=2):
    """
    Load all images from images_dir and return numpy array of grayscale images.
    """
    images = []
    for filename in os.listdir(images_dir):
        if 'jpeg' not in filename:
            print(f"skipping file: {filename}")
            continue

        image_path = os.path.join(images_dir, filename)
        print(image_path)
        img = cv2.imread(image_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)
        if len(images) >= num_images:
            break
    np_images = np.array(images)
    print(f"Loaded {len(images)} images; shape = {np_images.shape}")
    return np_images


def annotate(im, msg):
    clicks = []
    def click(event):
        x, y = event.xdata, event.ydata
        clicks.append([x, y])
        print(len(clicks), x, y)

    plt.title(msg)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(im, cmap='gray')

    _ = fig.canvas.mpl_connect('button_press_event', click)
    plt.show()
    return np.array(clicks)


def annotate_peak(images_arr):
    """
    Manually annotate peaks in each image.
    Click (top, bottom) of peak object.
    Return (2 x N), (2 x N) of peak coordinates.
    """
    peaks = []
    for img in images_arr:
        peak_coord = annotate(img, "Annotate peak: Click top and bottom of peak object.")
        peaks.append(peak_coord)
    assert len(peaks) == 2  # assume two images
    return peaks[0], peaks[1]


def annotate_prior(images_arr):
    """
    Manually annotate priors in each image.
    Click (top, bottom) of prior object.
    Return (2 x N), (2 x N) of prior coordinates.
    """
    prior = []
    for img in images_arr:
        prior_coord = annotate(img, "Annotate prior: Click top and bottom of prior object.")
        prior.append(prior_coord)
    assert len(prior) == 2  # assume two images
    return prior[0], prior[1]


def extract_kps_two_view(
        images_arr, 
        N=50, 
        corresp_path='corresp.npy', viz_match_path="viz_matches.jpg", 
        force_recompute=False
    ):
    """extract keypoints from two images"""
    if os.path.exists(corresp_path) and not force_recompute:
        pts1, pts2 = np.load(corresp_path)
        print(f"Loading correspondences from file {corresp_path}")
        print("pts1: ", pts1.shape)
        print("pts2: ", pts2.shape)
        return pts1, pts2

    print("extracting keypoints...")
    img1 = images_arr[0].squeeze()
    img2 = images_arr[1].squeeze()
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # create feature matcher
    print("computing correspondences...")
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    print(f"found {len(good)} correspondences total, keeping top {N}.")
    # sort matches by distance
    matches = sorted(good, key = lambda x:x.distance)
    # draw first 50 matches
    matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    
    # show the image
    cv2.imshow('image', matched_img)
    cv2.imwrite(viz_match_path, matched_img)

    pts1 = np.array([keypoints_1[m.queryIdx].pt for m in matches[:N]])
    pts2 = np.array([keypoints_2[m.trainIdx].pt for m in matches[:N]])
    print("pts1: ", pts1.shape)
    print("pts2: ", pts2.shape)
    np.save(corresp_path, [pts1, pts2])
    return pts1, pts2


def find_fundamental_matrix(match_pts1, match_pts2):
    """Estimates fundamental matrix """
    F, Fmask = cv2.findFundamentalMat(
        match_pts1, 
        match_pts2, 
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=0.1, 
        confidence=.99
    )
    print("F: ", F)
    return F, Fmask


def to_cross_product_matrix(vector):
    """Converts a 3D vector to a cross product matrix"""
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])


def fundamental_matrix_to_camera_matrices(F):
    """
    Computes the camera matrices P and P' from the fundamental matrix F for the uncalibrated case
    In the canonnical form, P1 = [I | 0] and P2 = [M | m]
    where M = SF and m is the epipole
    """
    # Find the epipoles
    # e2 is the left null space of F
    U, D, Vh = np.linalg.svd(F)
    e2 = Vh[-1]  # last row of V^T
    e2 = e2/e2[-1] # normalize

    S = to_cross_product_matrix(e2)
    M = S @ F
    
    P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    P2 = np.concatenate((M, e2.reshape(3, 1)), axis=1)

    print("P1: ", P1)
    print("P2: ", P2)
    return P1, P2


def triangulate(P1, P2, twod_pts1, twod_pts2):
    """
    :param P1: 3x4 camera matrix 1
    :param P2: 3x4 camera matrix 2
    :param two_pts1: 2xN points from image 1
    :param two_pts2: 2xN points from image 2
    """
    print(f"{twod_pts1.shape=} {twod_pts2.shape=} {P1.shape=} {P2.shape=}")
    threed_pts = cv2.triangulatePoints(P1, P2, twod_pts1, twod_pts2)
    return threed_pts


# def linear_ls_triangulation(u1, P1, u2, P2):
#     """
#     Triangulation via Linear-LS method
#     """
#     # build A matrix for homogeneous equation system Ax=0
#     # assume X = (x,y,z,1) for Linear-LS method
#     # which turns it into AX=B system, where A is 4x3, X is 3x1 & B is 4x1
#     A = np.array([u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1],
#                     u1[0] * P1[2, 2] - P1[0, 2], u1[1] * P1[2, 0] - P1[1, 0],
#                     u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2],
#                     u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1],
#                     u2[0] * P2[2, 2] - P2[0, 2], u2[1] * P2[2, 0] - P2[1, 0],
#                     u2[1] * P2[2, 1] - P2[1, 1],
#                     u2[1] * P2[2, 2] - P2[1, 2]]).reshape(4, 3)

#     B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
#                     -(u1[1] * P1[2, 3] - P1[1, 3]),
#                     -(u2[0] * P2[2, 3] - P2[0, 3]),
#                     -(u2[1] * P2[2, 3] - P2[1, 3])]).reshape(4, 1)

#     ret, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
#     return X.reshape(1, 3)


def calibrate(threed_peak_coords, threed_prior_coords, prior_height):
    """
    Calibrate the 3d coordinates of the peak object.
    """
    


if __name__ == '__main__':
    img_path = 'data/simon'
    corresp_path = 'data/simon/corresp.npy'
    viz_match_path = 'viz_matches.jpg'

    images = load_images(img_path, num_images=2) # (2, H, W)
    pts1, pts2 = extract_kps_two_view(
        images, 
        N=50, 
        corresp_path=corresp_path, 
        viz_match_path=viz_match_path, 
        force_recompute=False
    )
    F, _ = find_fundamental_matrix(pts1, pts2)
    P1, P2 = fundamental_matrix_to_camera_matrices(F)

    peak_coords1, peak_coords2 = annotate_peak(images)  # (2 x 1), (2 x 1) annotate single 3d point in two images
    prior_coords1, prior_coords2 = annotate_prior(images) # (2 x 2), (2 x 2) annotate two 3d points in two images
    threed_peak_coords = triangulate(P1, P2, peak_coords1, peak_coords2)
    threed_prior_coords = triangulate(P1, P2, prior_coords1, prior_coords2)

    prior_height = 0.5  # placeholder
    camera_altitude = 1.5 # placeholder
    
    calibrated_peak_coords = calibrate(threed_peak_coords, threed_prior_coords, prior_height)
    height = height_estimation(calibrated_peak_coords, camera_altitude)
    print(height)
    