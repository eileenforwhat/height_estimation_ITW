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
from collections import defaultdict
from typing import List
from exif import get_gps_data, gps_distance, get_direction_data, get_relative_translation, get_relative_rotation
from scipy.optimize import least_squares
import pickle as pkl

to_homogeneous = lambda pts: cv2.convertPointsToHomogeneous(pts).squeeze().T
from_homogeneous = lambda pts: cv2.convertPointsFromHomogeneous(pts.T).squeeze()

def load_images(images_dir, num_images=2, selected_imgfiles = None):
    """
    Load all images from images_dir and return numpy array of grayscale images.
    """
    images = []
    metadata = defaultdict(list)
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
            gps = get_gps_data(image_path)
            direction = get_direction_data(image_path)

            metadata['gps_locs'].append(gps)
            metadata['gps_dirs'].append(direction)

        if len(images) >= num_images:
            break
    np_images = np.array(images)
    print(f"Loaded {len(images)} images; shape = {np_images.shape}")
    return np_images, metadata


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
        if m.distance < 0.5*n.distance:
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



def normalize_for_fundamental(points: np.array):
    # Take points in image pixel coordinate space and map 
    # so that they are zero-centered with mean distance=sqrt(2)
    # print(f"{points.shape=}")
    assert len(points.shape) == 2 # Nx2
    t = points.mean(axis=(0,))
    s = np.sqrt(2)/np.linalg.norm(points-t, axis=1).mean()

    T = np.array([
        [s, 0, -s*t[0]], 
        [0, s, -s*t[1]], 
        [0, 0, 1]])
    return T


def find_fundamental_matrix(pts1, pts2, normalize_points=True):
    """Estimates fundamental matrix """
    F, Fmask = cv2.findFundamentalMat(
        pts1, 
        pts2, 
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=1e-10, 
        confidence=.99
    )
    print("F: ", F)
    return F


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
    # print(f"{twod_pts1.shape=} {twod_pts2.shape=} {P1.shape=} {P2.shape=}")
    threeD_pts = cv2.triangulatePoints(P1, P2, twod_pts1, twod_pts2)
    # print(f"{threeD_pts.shape=}")
    return threeD_pts.squeeze()


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


def calibrate_gps(uncalibrated_3d_coords: List[np.array], C1, C2, gps_distance):
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
    C1 = C1 / C1[-1]
    C2 = C2 / C2[-1]
    print(f"{C1=} {C2=}")
    uncalibrated_prior_length = np.linalg.norm(C1 - C2, ord=2) # length of prior in previous coordinate system
    scale_factor = gps_distance / uncalibrated_prior_length
    print(f'{uncalibrated_prior_length=} {gps_distance=} {scale_factor=}')
    for coords in uncalibrated_3d_coords:
        coords = coords / coords[-1]
        print(f"{coords=}")
        coords[:3] = coords[:3] * scale_factor
        calibrated_3d_coords.append(coords)
    
    return calibrated_3d_coords, scale_factor


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


def show_viz_3d(peaks, cameras):
    # visualize in 3D each point in peak, prior, camera, keypoints
    # each object is an np matrix of shape (4, n)
    from mpl_toolkits.mplot3d import Axes3D
    cameras = cameras.T
    assert peaks.shape[0] == 4, f"Expected shape (4, n), got {peaks.shape}"
    assert cameras.shape[0] == 4, f"Expected shape (4, n), got {cameras.shape}"
    peaks = peaks / peaks[-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(peaks[0], peaks[1], peaks[2], c='r', marker='o', s=10)
    # ax.scatter(cameras[0], cameras[1], cameras[2], c='b', marker='o', s=20)
    ax.plot(peaks[0], peaks[1], peaks[2], c='r')
    ax.plot(cameras[0], cameras[1], cameras[2], c='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-150, 150)

    ax.legend(['peaks', 'cameras'])
    plt.show()


def calibrate_ground_plane(P1, P2, img1, img2, ground_plane_path, scale, force_recompute=False):
    if os.path.exists(ground_plane_path) and not force_recompute:
        plane = np.load(ground_plane_path)
        print(f"Loading ground plane from file {ground_plane_path}")
        print("ground plane: ", plane)
        return plane
    
    ground_pts = []
    for img in [img1, img2]:
        coords = annotate(img, "Calibrate ground plane: click on THREE points on ground plane.")
        ground_pts.append(coords)
    assert len(ground_pts) == 2  # assume two images
    threed_ground_pts = triangulate(P1, P2, ground_pts[0].T, ground_pts[1].T)  # 4,3
    plane = find_plane_equation(threed_ground_pts[:, 0]*scale, threed_ground_pts[:, 1]*scale, threed_ground_pts[:, 2]*scale)
    print("ground plane: ", plane)
    np.save(ground_plane_path, plane)
    print(f"Saved ground plane to file {ground_plane_path}")
    return plane


def find_plane_equation(p1, p2, p3):
    if p1.shape[0] == 4:
        p1 = p1 / p1[-1]
        p1 = p1[:3]
        p2 = p2 / p2[-1]
        p2 = p2[:3]
        p3 = p3 / p3[-1]
        p3 = p3[:3]
    # Create vectors from points
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)

    # Compute the cross product to find the normal
    normal = np.cross(v1, v2)

    # Extract coefficients
    a, b, c = normal
    d = -np.dot(normal, p1)

    return a, b, c, d


def height_estimation_from_ground(ground_plane, peak_coords):
    # ground_plane: (4, 1)
    # peak_coords: (4, 1)
    num = np.abs(ground_plane[0] * peak_coords[0] + ground_plane[1] * peak_coords[1] + ground_plane[2] * peak_coords[2] + ground_plane[3])
    denom = np.sqrt(ground_plane[0] ** 2 + ground_plane[1] ** 2 + ground_plane[2] ** 2)
    return num / denom


def enforce_det_0(L):
    U, S, VT = np.linalg.svd(L)
    S[-1] = 0
    return U @ np.diag(S) @ VT
def enforce_postive_definite(L):
    # U, S, VT = np.linalg.svd(L)
    # return U @ np.diag(np.abs(S)) @ VT
    return np.sqrt(L.T @ L)

def K_from_L(L):
    # L = enforce_det_0(L)
    L = enforce_postive_definite(L[:3, :3])
    return np.linalg.cholesky(L).T

def v_from_L(L, K):
    Kv = L[:3, 3]
    # solve for v given K @ v = L[:3,3]
    v = np.linalg.solve(K, Kv)
    return v

def H_from_K_v(K, v):
    # H: 4x4
    # K: 3x3
    # v: 3
    # H = [[K | 0], [v.T | 1]]
    H1 = np.concatenate((K, np.zeros((3, 1))), axis=1)
    H2 = np.concatenate((v.reshape((1,3)), np.ones((1, 1))), axis=1)
    H = np.concatenate((H1, H2), axis=0)
    return H

"""
result = least_squares(calculate_error, initial_params, args=(initial_3d_points, known_scale_distance))

# Extract optimized camera poses and 3D points
optimized_params = result.x
optimized_camera_poses = optimized_params[:len(initial_camera_poses) * 6].reshape(initial_camera_poses.shape)
optimized_3d_points = optimized_params[len(initial_camera_poses) * 6:].reshape(initial_3d_points.shape)
"""
def bundle_adjustment_with_constraints(
    x3d, P1, R_rel, t_rel, pts1, pts2, peak_coords1, peak_coords2, 
    params_path='data/cathedral/bundle.pkl', force_recompute=False, auto_calibration=False):
    # X must be a vector of shape (n,)
    # K1, K2, x3d: initial params, will update
    # R_rel, t_rel, pts1, pts2: known_params, constant.
    # Proj1 = K1@[I|0], Proj2 = K2@[R|t]
    # Assume R_rel and t_rel are fixed and known for sure. we do not update them.
    if os.path.exists(params_path) and not force_recompute:
        print(f"Loading bundle adjustment from file {params_path}")
        return pkl.load(open(params_path, 'rb'))
    
    
    K_from_params = lambda f_x, f_y, c_x, c_y: np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    L_from_params = lambda l1, l2, l3, l4, l5, l6, l7, l8, l9: np.array([[l1, l2, l3, l4], [l2, l5, l6, l7], [l3, l6, l8, l9], [l4, l7, l9, 1]])

    def calculate_error(params: np.ndarray, R_rel, t_rel, pts1, pts2, peak_coords1, peak_coords2, **kwargs):
        assert len(params.shape) == 1 # (n,)
        errors = []
        
        K1, K2 = K_from_params(*params[0:4]), K_from_params(*params[4:8])
        L = L_from_params(*params[8:17]) # (4, 4)
        x3d = params[17:].reshape((-1, 4)) # (N, 4)
        
        if auto_calibration:
            K1 = K2 = K_from_L(L)
            
        
        P1 = K1 @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        P2 = K2 @ np.concatenate((R_rel, t_rel), axis=1)
        x2d1 = (P1 @ x3d.T).T # (N, 3)
        x2d2 = (P2 @ x3d.T).T
        
        # calculate reprojection error between x2d1 and pts1, and between x2d2 and pts2
        reprojection_error1 = np.linalg.norm(normalize_for_fundamental(x2d1) - normalize_for_fundamental(pts1), axis=0)
        reprojection_error2 = np.linalg.norm(normalize_for_fundamental(x2d2) - normalize_for_fundamental(pts2), axis=0)
        # print(f"{x2d1.shape=}  {pts1.shape=}  {reprojection_error1.shape=}")
        
        # square pixels constraint
        aspect_ratio_error = np.array([K1[0,0] - K1[1,1]])
        
        # Same K1 and K2 constraint
        same_K_error = (K1 - K2).reshape((-1,))
        
        # Peak vertical constraint: peak_coords1 and peak_coords2 should have same x and z coords
        triangulated_peak_coords = triangulate(P1, P2, peak_coords1.T, peak_coords2.T)  # (4 x 2) 3d points
        peak_vertical_error = np.array([triangulated_peak_coords[0, 0] - triangulated_peak_coords[0, 1]])**2 + \
                              np.array([triangulated_peak_coords[2, 0] - triangulated_peak_coords[2, 1]])**2
        
        
        errors += [reprojection_error1, reprojection_error2, aspect_ratio_error, same_K_error]
        
        # Auto Calibration
        if auto_calibration:
            # w1 = K1 @ K1.T
            # w2 = K2 @ K2.T
            w1 = P1 @ L @ P1.T
            w2 = P2 @ L @ P2.T
            
            # zero skew, equal focal lengths, shared intrinsics
            w_skew_error = np.array([w1[0,1]* w1[2,2] - w1[0,2] * w1[1,2]])
            w_equal_focal_error = np.array([(w1[0,0] - w1[1,1]) * w1[2,2] - (w1[0,2]**2 - w1[1,2]**2)])
            ratio = w1[2,2] / w2[2,2]
            w_shared_intrinsics_error = np.array([
                (ratio - w1[0,0] / w2[0,0])**2 + 
                (ratio - w1[1,1] / w2[1,1])**2 + 
                (ratio - w1[0,1] / w2[0,1])**2 + 
                (ratio - w1[0,2] / w2[0,2])**2 + 
                (ratio - w1[1,2] / w2[1,2])**2
            ])
            errors += [w_skew_error, w_equal_focal_error, w_shared_intrinsics_error]
        
        
        result = np.concatenate(errors) 
        assert len(result.shape) == 1 # (m,)
        return result
    

    # Initialize parameters
    # we have known Proj1 and Proj2. How do we initialize K1, K2?
    K1_params = K2_params = (P1[0,0], P1[1,1], P1[0,2], P1[1,2])
    L_params = np.ones((9,))
    initial_params = np.concatenate([K1_params, K2_params, L_params, x3d.reshape((-1,))])
    
    # Run least squares solver
    res_wrapped = least_squares(calculate_error, initial_params, args=(R_rel, t_rel, pts1, pts2, peak_coords1, peak_coords2))
    
    # return optimized params
    # K1, K2 = K_from_params(*res_wrapped.x[0:4]), K_from_params(*res_wrapped.x[4:8])
    K1, K2 = K_from_params(*res_wrapped.x[0:4]), K_from_params(*res_wrapped.x[4:8])
    L = L_from_params(*res_wrapped.x[8:17]) # (4, 4)
    x3d = res_wrapped.x[17:].reshape((-1, 4)) # (N, 4)
    
    if auto_calibration:
        L = L_from_params(*res_wrapped.x[:9]) # (4, 4)
        K1 = K2 = K_from_L(L)
        v = v_from_L(L, K1)
        H = H_from_K_v(K1, v)
        P1 = K1 @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1) @ H
        P2 = K2 @ np.concatenate((R_rel, t_rel), axis=1) @ H
    else:
        P1 = K1 @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        P2 = K2 @ np.concatenate((R_rel, t_rel), axis=1)
    
    pkl.dump((K1, K2, x3d, P1, P2), open(params_path, 'wb'))
    
    return K1, K2, x3d, P1, P2
    

if __name__ == '__main__':
    img_path = 'data/cathedral'
    corresp_path = 'data/cathedral/corresp.npy'
    manual_corresp_path = 'data/cathedral/corresp_manual.npy'
    viz_match_path = 'data/cathedral/viz_matches.jpg'
    selected_imgfiles=['IMG_6350.jpeg', 'IMG_6348.jpeg']
    
    images, metadata = load_images(
        img_path, 
        num_images=2, 
        selected_imgfiles=selected_imgfiles
    ) # (2, H, W)
    gps_locations = metadata['gps_locs']
    gps_directions = metadata['gps_dirs']
    pts1, pts2 = extract_kps_two_view(
        images, 
        N=200, 
        corresp_path=corresp_path, 
        viz_match_path=viz_match_path, 
        force_recompute=False
    )
    # pts1, pts2 = manual_extract_kps_two_view(
    #     images, 
    #     corresp_path=manual_corresp_path, 
    #     force_recompute=False
    # )
    
    # annotate top and bottom of object
    peak_coords1, peak_coords2 = annotate_peak(
        images, 
        peaks_path='data/cathedral/peak_coords.npy',
        force_recompute=False,
        msg="Annotate peak: Click top and bottom of peak object."
    )  # (2 x 1), (2 x 1) annotate single 3d point in two images
    
    
    T1, T2 = map(normalize_for_fundamental, [pts1, pts2])
    npts1 = T1 @ to_homogeneous(pts1)
    npts2 = T2 @ to_homogeneous(pts2)
    npts1, npts2 = map(from_homogeneous, [npts1, npts2])
    
    F = find_fundamental_matrix(npts1, npts2)
    # F = T2.T @ F @ T1 # Unnormalize: Transform solution to pixel space
    
    inliers1, inliers2 = find_inliers(npts1, npts2, F)
    # show_viz(images[0], images[1], inliers1, inliers2, F)
    
    # exists projective ambiguity (i.e., Ambiguity up to a homography H)
    # (P, P’) is a solution -> (PH, P’H) is a solution
    P1, P2 = fundamental_matrix_to_camera_matrices(F) 
    
    inliers_3d = triangulate(P1, P2, from_homogeneous(inliers1.T).T, from_homogeneous(inliers2.T).T).T  # (N, 4)
    theta_rel, R_rel = get_relative_rotation(*gps_directions)
    t_rel, magnitude_rel = get_relative_translation(*gps_locations[0], *gps_locations[1])
    K1, K2, x3d, P1, P2 = bundle_adjustment_with_constraints(
        inliers_3d, P1, R_rel, t_rel, inliers1, inliers2, peak_coords1, peak_coords2, force_recompute=False
    )
    print(f"{K1=}\n {K2=}\n {x3d.shape=}\n {P1=}\n {P2=}")
    C1, C2 = get_camera_centers([P1, P2])
    
    threeD_peak_coords = triangulate(P1, P2, peak_coords1.T, peak_coords2.T)  # (4 x 2) 3d points
    # show_viz_3d(threeD_peak_coords, np.array([C1, C2]))
    gps_distance = gps_distance(*gps_locations[0], *gps_locations[1])
    print(f'{gps_distance=}')
    calibrated_coords, scale_factor = calibrate_gps([threeD_peak_coords, C1, C2], C1, C2, gps_distance)
    calibrated_peak_coords, C1, C2 = calibrated_coords

    print(f'{C1=} {C2=}')
    print(f'{calibrated_peak_coords=}')
    print(f'{scale_factor=}')
    # show_viz_3d(calibrated_peak_coords, np.array([C1, C2]))
    height = height_estimation(calibrated_peak_coords, camera_altitude=None)
    # ground_plane = calibrate_ground_plane(P1, P2, images[0], images[1], 'data/simon/ground_plane.npy', scale_factor, force_recompute=True)
    # height = height_estimation_from_ground(ground_plane, calibrated_peak_coords[:, 0])
    print(f'{height=}')
    