# Two view reconstruction (w/o manual annots)
# 1. keypoint extraction and corresponding (SIFT) - done
# 2. Compute F - done
# 3. Compute P, P' from F - done
# 4. manually annotate peak - done
# 5. triangulate for the peak to get height - simon
# 6. calibrate the height - eileen
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def load_images(images_dir, num_images=2, selected_imgfiles = None):
    """
    Load all images from images_dir and return numpy array of grayscale images.
    """
    images = []
    for filename in os.listdir(images_dir):
        if 'jpeg' not in filename:
            print(f"skipping file: {filename}")
            continue

        if selected_imgfiles and filename in selected_imgfiles:
            image_path = os.path.join(images_dir, filename)
            print(f"loading from {image_path}")
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


def annotate_peak(
        images_arr, 
        peaks_path, 
        force_recompute=False, 
        msg="Annotate peak: Click top and bottom of peak object."
    ):
    """
    Manually annotate peaks in each image.
    Click (top, bottom) of peak object.
    Return (2 x N), (2 x N) of peak coordinates.
    """
    if os.path.exists(peaks_path) and not force_recompute:
        peaks1, peaks2 = np.load(peaks_path)
        peaks1 = np.array(peaks1).reshape(2, -1)
        peaks2 = np.array(peaks2).reshape(2, -1)
        print(f"Loading peaks from file {peaks_path}")
        print("peaks1: ", peaks1)
        print("peaks2: ", peaks2)
        return peaks1, peaks2
    
    peaks = []
    for img in images_arr:
        peak_coord = annotate(img, msg)
        peaks.append(peak_coord)
    assert len(peaks) == 2  # assume two images
    np.save(peaks_path, peaks)
    return peaks[0], peaks[1]


def manual_extract_kps_two_view(
        images_arr, 
        N=8, 
        corresp_path='corresp_manual.npy', 
        force_recompute=False
    ):
    """
    Returns pts1, pts2: (N x 2) coordinates of keypoints in each image
    """

    if os.path.exists(corresp_path) and not force_recompute:
        pts1, pts2 = np.load(corresp_path)
        pts1 = np.array(pts1).reshape(-1, 2)
        pts2 = np.array(pts2).reshape(-1, 2)
        print(f"Loading correspondences from file {corresp_path}")
        print("pts1: ", pts1.shape)
        print("pts2: ", pts2.shape)
        return pts1, pts2
    
    pts = [[], []] # 2xNx2
    for j in range(N):
        print('annotations for correspondence ', j)
        for i, img in enumerate(images_arr):
            coord = annotate(img, "Annotate 1 correspondence")
            pts[i].append(coord)

    print(len(pts[0]), len(pts[1]))
    print(pts[0])
    np.save(corresp_path, pts)
    print("saving correspondences to file ", corresp_path)
    return pts[0], pts[1]


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


def visualize_correspondences(img1, img2, pts1, pts2):
    # draw first 50 matches on a side by side image
    img = np.hstack((img1, img2))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for _, p1, p2 in zip(range(50), pts1.T, pts2.T):
        p1 = p1.squeeze()
        p2 = p2.squeeze()
        p2[0] += img1.shape[1]
        cv2.circle(img, tuple(p1), 3, (0, 255, 0), -1)
        cv2.circle(img, tuple(p2), 3, (0, 255, 0), -1)
        # cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 1)
    cv2.imshow('image', img)

def find_fundamental_matrix(match_pts1, match_pts2):
    """Estimates fundamental matrix """
    F, Fmask = cv2.findFundamentalMat(
        match_pts1, 
        match_pts2, 
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=1, 
        confidence=.99
    )
    print("F: ", F)
    return F, Fmask

def find_inliers(match_pts1, match_pts2, F):
    inliers = [[], []]
    for p1, p2 in zip(match_pts1, match_pts2):
        p1 = np.append(p1, 1)
        p2 = np.append(p2, 1)
        err = np.abs(p2 @ F @ p1)
        if err < 0.1:
            inliers[0].append(p1)
            inliers[1].append(p2)
    return np.array(inliers[0]), np.array(inliers[1])


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
    threeD_pts = cv2.triangulatePoints(P1, P2, twod_pts1, twod_pts2)
    print(f"{threeD_pts.shape=}")
    return threeD_pts


def get_camera_centers(camera_matrices: List[np.array]):
    """
    :param camera_matrices: list of 3x4 camera matrices
    :return: list of 4x1 camera centers
    """
    camera_centers = []
    for P in camera_matrices:
        # get camera centers by solving PC = 0 for C using SVD
        U, D, Vh = np.linalg.svd(P)
        camera_center = Vh[-1]
        camera_center = camera_center / camera_center[-1]
        camera_centers.append(camera_center)
    print(f"{camera_centers[0].shape=}")
    assert camera_centers[0].shape == (4,)
    return camera_centers


def calibrate(uncalibrated_3d_coords: List[np.array], threeD_prior_coords, prior_length):
    """
    Calibrate the 3d coordinates of the peak object.
    given the actual length of the prior object, rescale the 3D points,
    
    :param uncalibrated_3d_coords: List[4xn] Homogeneous 3D coordinates of the peak object from two views, and the camera centers
    :param threeD_prior_coords: 4x2 Homogeneous 3D coordinates of the prior object from two views
    :param prior_length: actual length of the prior object
    :return: 4x2 calibrated 3D coordinates of the peak object from two views
    """
    calibrated_3d_coords = []
    # change from homogeneous to heterogenous coordinates
    threeD_prior_coords = threeD_prior_coords / threeD_prior_coords[-1]
    print(f"{threeD_prior_coords=}")
    assert threeD_prior_coords.shape == (4, 2)
    uncalibrated_prior_length = np.linalg.norm(threeD_prior_coords[:,0] - threeD_prior_coords[:,1], ord=2) # length of prior in previous coordinate system
    scale_factor = prior_length / uncalibrated_prior_length
    print(f'{uncalibrated_prior_length=} {prior_length=} {scale_factor=}')
    for coords in uncalibrated_3d_coords:
        coords = coords / coords[-1]
        print(f"{coords=}")
        coords[:3] = coords[:3] * scale_factor
        calibrated_3d_coords.append(coords)
    return calibrated_3d_coords


def height_estimation(calibrated_peak_coords, camera_altitude=None):
    """
    Estimate the height of the peak object.
    :param calibrated_peak_coords: 4x2 calibrated 3D coordinates of the peak object from two views
    :param camera_altitude: altitude of the camera
    :return: height of the peak object
    """
    # change from homogeneous to heterogenous coordinates
    calibrated_peak_coords = calibrated_peak_coords / calibrated_peak_coords[-1]

    # compute the height of the peak object in relation to the camera position
    if camera_altitude is None:
        height = np.linalg.norm(calibrated_peak_coords[:,0] - calibrated_peak_coords[:,1], ord=2)

    return height


def show_viz(img1, img2, pts1, pts2, F, title=""):
    if pts1.shape[1] == 2:
        pts1 = np.append(pts1, np.ones((len(pts1), 1)), 1)
    if pts2.shape[1] == 2:
        pts2 = np.append(pts2, np.ones((len(pts2), 1)), 1)
    num_pts = pts1.shape[0]
    colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
    colors = np.tile(colors, int(num_pts / len(colors)) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax1.scatter(pts1[:, 0], pts1[:, 1], c=colors[:num_pts])
    ax1.axis('off')

    ax2.imshow(img2)
    ax2.scatter(pts2[:, 0], pts2[:, 1], c=colors[:num_pts])
    ax2.axis('off')

    lines = []
    for i in range(num_pts):
        lines.append(F @ pts1[i])
    
    for i, (a, b, c) in enumerate(lines):
        x = np.array([0, img2.shape[1]])
        y = (-c - a * x) / b
        ax2.plot(x, y, c=colors[i % len(colors)], )
    ax2.set_xlim(0, img1.shape[1])
    ax2.set_ylim(img1.shape[0], 0)
    plt.savefig('output/' + title + '.png')
    plt.show()


def show_viz_3d(peaks, priors, cameras):
    # visualize in 3D each point in peak, prior, camera, keypoints
    # each object is an np matrix of shape (4, n)
    from mpl_toolkits.mplot3d import Axes3D
    cameras = cameras.T
    assert peaks.shape[0] == 4, f"Expected shape (4, n), got {peaks.shape}"
    assert priors.shape[0] == 4, f"Expected shape (4, n), got {priors.shape}"
    assert cameras.shape[0] == 4, f"Expected shape (4, n), got {cameras.shape}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(peaks[0], peaks[1], peaks[2], c='r', marker='o', s=1)
    ax.scatter(priors[0], priors[1], priors[2], c='b', marker='o', s=2)
    # ax.scatter(cameras[0], cameras[1], cameras[2], c='g', marker='o', s=5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend(['peaks', 'priors', 'cameras'])
    plt.show()
    

if __name__ == '__main__':
    img_path = 'data/simon'
    corresp_path = 'data/simon/corresp.npy'
    manual_corresp_path = 'data/simon/corresp_manual.npy'
    viz_match_path = 'viz_matches.jpg'

    images = load_images(
        img_path, 
        num_images=2, 
        # selected_imgfiles=['IMG_6179.jpeg', 'IMG_6182.jpeg']
        selected_imgfiles=['2.jpeg', '3.jpeg']
    ) # (2, H, W)
    pts1, pts2 = extract_kps_two_view(
        images, 
        N=300, 
        corresp_path=corresp_path, 
        viz_match_path=viz_match_path, 
        force_recompute=False
    )
    # pts1, pts2 = manual_extract_kps_two_view(
    #     images, 
    #     corresp_path=manual_corresp_path, 
    #     force_recompute=False
    # )
    F, _ = find_fundamental_matrix(pts1, pts2)
    inliers1, inliers2 = find_inliers(pts1, pts2, F)
    show_viz(images[0], images[1], inliers1, inliers2, F)
    
    # exists projective ambiguity (i.e., Ambiguity up to a homography H)
    # (P, P’) is a solution -> (PH, P’H) is a solution
    P1, P2 = fundamental_matrix_to_camera_matrices(F) 
    C1, C2 = get_camera_centers([P1, P2])
    

    peak_coords1, peak_coords2 = annotate_peak(
        images, 
        peaks_path='data/simon/peak_coords.npy',
        force_recompute=False,
        msg="Annotate peak: Click top and bottom of peak object."
    )  # (2 x 1), (2 x 1) annotate single 3d point in two images
    prior_coords1, prior_coords2 = annotate_peak(
        images, 
        peaks_path='data/simon/prior_coords.npy',
        force_recompute=False,
        msg="Annotate prior: Click top and bottom of prior object."
    ) # (2 x 2), (2 x 2) annotate two 3d points in two images
    threeD_peak_coords = triangulate(P1, P2, peak_coords1, peak_coords2)  # (4 x 2) 3d points
    threeD_prior_coords = triangulate(P1, P2, prior_coords1, prior_coords2) # (4 x 2) 3d points

    prior_height = 175  # placeholder, units: centimeters (3d)
    camera_altitude = None # None if we annotate two points (peak, base) of the mountain
    
    show_viz_3d(threeD_peak_coords, threeD_prior_coords, np.array([C1, C2]))

    calibrated_peak_coords, calibrated_prior_coords, C1, C2 = calibrate([threeD_peak_coords, threeD_prior_coords, C1, C2], threeD_prior_coords, prior_height)
    show_viz_3d(calibrated_peak_coords, calibrated_prior_coords, np.array([C1, C2]))
    height = height_estimation(calibrated_peak_coords, camera_altitude)
    print(height)
    assert 100 < height < 200, f"Simon's height out of bounds. Got {height=}. Expected 175cm."
    