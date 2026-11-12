from initSettings import Cameras
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi


def closest_point_to_lines(points, dirs):
    """points, dirs: shape (N, 3). Each line is points[i] + t*dirs[i]."""
    A = np.asarray(points, float)
    D = np.asarray(dirs, float)

    I = np.eye(3)
    M = I[None] - D[:, :, None] * D[:, None, :]        # (N,3,3) projectors
    S = M.sum(axis=0)
    b = (M @ A[:, :, None]).sum(axis=0).ravel()

    p = np.linalg.solve(S, b)

    return p

def model():
    Locs = []

    idx = 0
    for i in range(10):
        print(f"\n\nidx = {idx}")
        offsets = []
        slopes = []
        for camNo, cam in Cameras.items():
            # phi = degToRad(cam.getHeightOr())
            # theta = degToRad(cam.getCompOr())
            frame = cam.getFrameNo(idx)

            print("")
            print("-" * 20)
            print(f"getting the line for camera No. {camNo}")
            # print(f"camLoc = ({cam.getX()}, {cam.getY()}, {cam.getZ()}), compOr = {theta}, heightOr = {phi}")
            print(f"frame = {frame}")


            camOffset = np.array([cam.getX(), cam.getY(), cam.getZ()])
            camSlope = cam.getCamSlope()

            xCamBase = cam.getXCamBase()
            yCamBase = cam.getYCamBase()

            offset = camOffset + xCamBase * frame[0] + yCamBase * frame[1]

            offsets.append(offset)
            slopes.append(camSlope)

        print("#" * 20)
        print(f"offsets = {offsets}\nslopes = {slopes}")

        loc = closest_point_to_lines(offsets, slopes)

        print(f"loc = {loc}")
        Locs.append(loc)

        idx += 1

    print(Locs)
    for i in range(len(Locs)):
        Locs[i] = np.round(Locs[i], 3)


    P = np.vstack(Locs)          # shape (N, 3)
    x, y, z = P.T                  # or P[:,0], P[:,1], P[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, marker='o')   # use ax.scatter(x, y, z) for points only
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.show()