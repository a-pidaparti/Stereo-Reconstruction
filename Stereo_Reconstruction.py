import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

RANSAC_ITER = 5000
RANSAC_THRESH = .1
RATIO_THRESH = .5
DENSE_SIZE = 16

## Done
def find_match(img1, img2):
    sift1 = cv2.SIFT_create()
    sift2 = cv2.SIFT_create()

    kp1, d1 = sift1.detectAndCompute(img1, None)
    kp2, d2 = sift2.detectAndCompute(img2, None)

    nbrs_left = NearestNeighbors(n_neighbors=2).fit(d2)
    distances_left, indices_left = nbrs_left.kneighbors(d1)

    nbrs_right = NearestNeighbors(n_neighbors=2).fit(d1)
    distances_right, indices_right = nbrs_right.kneighbors(d2)

    pts1, pts2 = [], []
    for index, value in enumerate(distances_left):
        ## Ratio test
        ratio_left = value[0]/value[1]
        corresponding_pt_ind = indices_left[index, 0]
        value_right = distances_right[corresponding_pt_ind]
        ratio_right = value_right[0]/value_right[1]
        if ratio_left < RATIO_THRESH and ratio_right < RATIO_THRESH:
            pts1.append(kp1[index].pt)
            pts2.append(kp2[indices_left[index, 0]].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    return pts1, pts2

## Done
def compute_F(pts1, pts2):
    ## I think core idea is use RANSAC in the following way. A single iteration selects 8 points, computes F, and
    ## computes error (distance from 0 for each point in pts1 and pts2) and selects F with lowest error
    try:
        best_F = sio.loadmat('best_f.mat')['F']
        return best_F
    except:
        pass
    best_inliers = 0
    best_F = np.zeros((3,3))
    for not_used in range(RANSAC_ITER):
        choices = np.random.choice(range(pts1.shape[0]), 8, replace=False)
        pts1_choices = pts1[choices]
        pts2_choices = pts2[choices]
        A = np.zeros(shape=(8,9))
        for pt in range(8):
            u = pts1_choices[pt]
            v = pts2_choices[pt]
            A[pt, 0] = u[0] * v[0]
            A[pt, 1] = u[1] * v[0]
            A[pt, 2] = v[0]
            A[pt, 3] = u[0] * v[1]
            A[pt, 4] = u[1] * v[1]
            A[pt, 5] = v[1]
            A[pt, 6] = u[0]
            A[pt, 7] = u[1]
            A[pt, 8] = 1

        F_unshaped = null_space(A)[:, 0]
        F = F_unshaped.reshape((3, 3))
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ Vt
        num_inliers = 0
        for pt in range(pts1.shape[0]):
            v = np.array([pts1[pt, 0], pts1[pt, 1], 1]) ## v is already transposed
            u = np.array([pts2[pt, 0], pts2[pt, 1], 1])
            if abs(v @ F.T @ u.T) < RANSAC_THRESH and abs(u @ F @ v.T) < RANSAC_THRESH:
                num_inliers += 1
        if num_inliers > best_inliers:
            best_F = F
            best_inliers = num_inliers
    sio.savemat('best_f1.mat', mdict={'F':best_F})
    return best_F

## Done
def get_skew_symmetric(pt3d):
    SS = np.zeros((3,3))
    SS[1,0] = pt3d[2]
    SS[2,0] = -1 * pt3d[1]
    SS[0,1] = -1 * pt3d[2]
    SS[2,1] = pt3d[0]
    SS[0,2] = pt3d[1]
    SS[1,2] = -1 * pt3d[0]
    return SS


def triangulation(P1, P2, pts1, pts2):
    pts3D = np.zeros((pts1.shape[0], 3))
    for ind in range(pts1.shape[0]):
        u = pts1[ind]
        v = pts2[ind]
        uSS = get_skew_symmetric(np.array([u[0], u[1], 1]))
        vSS = get_skew_symmetric(np.array([v[0], v[1], 1]))
        uP1 = uSS @ P1
        vP2 = vSS @ P2
        stacked = np.vstack((uP1[:2], vP2[:2]))
        U, S, Vt = np.linalg.svd(stacked)
        S[-1] = 0
        stacked = U @ np.diag(S) @ Vt
        pt3D = null_space(stacked)[:, 0]
        pt3D /= pt3D[3]
        pts3D[ind, :] = pt3D[:3].reshape((3))
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    best_C = Cs[0]
    best_R = Rs[0]
    best_pts3D = pts3Ds[0]
    best_inliers = -1
    inliers = []
    for index in range(len(Rs)):
        num_inliers = 0
        R = Rs[index]
        C = Cs[index]
        r3t = R[2, :]
        for point in pts3Ds[index]:
            cheirality_c1 = np.dot(r3t, (point - C[:, 0]))

            ## Other camera is defined to be origin, rotation matrix = I, r3t = [0,0,1]
            cheirality_c2 = np.dot(np.array([0,0,1]), (point - C[:, 0]))
            if cheirality_c1 > 0 and cheirality_c2 > 0:
                num_inliers += 1
        inliers += [num_inliers]
        if num_inliers > best_inliers:
            best_C = C
            best_R = R
            best_pts3D = pts3Ds[index]
            best_inliers = num_inliers
    return best_R, best_C, best_pts3D


def compute_rectification(K, R, C):
    rx = (C / np.linalg.norm(C))[:, 0].T
    rz_tilde = np.array([0,0,1]).T
    rz_num = rz_tilde - (rz_tilde.dot(rx.T) * rx.T)
    rz = rz_num / np.linalg.norm(rz_num)

    ry = np.cross(rz, rx.T)
    R_rect = np.zeros((3,3))
    R_rect[0,:] = rx
    R_rect[1,:] = ry.T
    R_rect[2, :] = rz.T

    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)
    return H1, H2


def dense_match(img1, img2):
    kp1 = []
    kp2 = []
    sift = cv2.SIFT_create()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            kp1.append(cv2.KeyPoint(j, i, DENSE_SIZE))

    _, img1_dense = sift.compute(img1, kp1)
    sift = cv2.SIFT_create()
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            kp2.append(cv2.KeyPoint(j, i, DENSE_SIZE))
    _, img2_dense = sift.compute(img2, kp2)

    shape = (img1.shape[0], img1.shape[1], 128)
    img1_dense = img1_dense.reshape(shape)
    img2_dense = img2_dense.reshape(shape)
    disparity = np.zeros(shape=(img1_dense.shape[0], img1.shape[1]))
    for i in range(disparity.shape[0]):
        ## Core idea- nearest neighbor in a row will minimize the disparity function because
        ## those two pixels are defined to have the smallest difference in feature space
        ## ** Idea given by Eric Heidal, TA **
        nbrs = NearestNeighbors(n_neighbors=1).fit(img1_dense[i, :, :])
        _, indices = nbrs.kneighbors(img2_dense[i, :, :])

        ## get coordinates of left image row, then subtract from indices in right image
        img1_x_cor = range(img1.shape[1])
        diff = indices[:, 0] - img1_x_cor

        ## Zero any pixel that has a value of 0 in img1
        zero_indices = np.argwhere(img1[i, :] <= 0)
        diff[zero_indices] = 0
        disparity[i, :] = diff
    return disparity

# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3,1))
    p1, p2 = (0, int(-el[2] / el[1])), (img_width, int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_pose_with_pts(R, C, pts3d):
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    R2, C2, pts3D = R, C, pts3d
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    draw_camera(ax, R1, C1, 5)
    draw_camera(ax, R2, C2, 5)
    ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
    set_axes_equal(ax)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)
    visualize_camera_pose_with_pts(R, C, pts3D)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_vis = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2RGB)
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)



    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)
    #
    # # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
