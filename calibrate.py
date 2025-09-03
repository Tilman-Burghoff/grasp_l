# idea:
# sample n number of points (arUco marker centers) from fixed camera position set by user
# find ground truth positions by moving the gripper to the point, use gripper frame.
# then solve argmin_theta sum_(z_cam, z_true) ||T_theta * z_true - z_cam|| st. ||q|| = 1, where theta = (t, q)\in R^7
# is the transformation vector consisting of the translation t and rotation q as a normalized quaternion.
# T is the homogenous transformation matrix T = [[R_q,t], [0,1]], where R_q is the rotation corresponding to the quaternion
# TODO: either solve for theta completly (ie start with cameraframe = gripperframe)
# or use the existing calibration and only compute "residual error"
# advantages of "starting fresh": easier transformation, since we dont have to figure out how to account for existing calibration
# and the calibration provides a good baseline/sanity check for our new calibration
# disatvantages: we might want to run the calibration multiple times to incrementally increase the quality.
# but we could also just iterate the dataset creation longer, ie look at the same markers from different poses

from aruco_tracker import Smoothing, ArucoTracker
import robotic as ry
import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation as R

C = ry.config()
bot = ry.BotOp(C, True)
tracker = ArucoTracker(C)
rgb, _, points = bot.getImageDepthPcl()
pcl = C.addFrame('pcl', 'l_cameraWrist')
pcl.setPointCloud(points, rgb)

iters = int(input("Number of iterations:"))
visited = set()

measurements = []
for i in range(iters):
    print("Move bot into position, press q when done")
    bot.hold(floating=True, damping=False)
    while bot.getKeyPressed() != 'q':
        rgb, depth, points = bot.getImageDepthPcl('l_cameraWrist')
        tracker.track_markers(rgb, points)
        tracker.tag_markers()
        bot.sync(C, viewMsg="Move bot such that markers are visible, then press q.")

    bot.hold(floating=False, damping=True)
    C.sync(False)
    q = C.getJointState()
    print("Collecting measurements")
    tracker.position_smoothing = Smoothing.LAST
    tracker.reset_average()
    for i in range(10):
        rgb, depth, points = bot.getImageDepthPcl('l_cameraWrist')
        visible = tracker.track_markers(rgb, points)
        tracker.tag_markers()
        bot.sync(C, .1, "Collecting measurements")
        tracker.position_smoothing = Smoothing.AVERAGE

    visited.update(set(visible))
    measurements.append((q, {id: tracker[id] for id in visible}))


print("Gathering ground truth")
ground_truth = {}
for id in visited:
    print(f"Move gripper to marker {id}, then press q")
    bot.hold(floating=True, damping=False)

    while bot.getKeyPressed() != 'q':
        bot.sync(C, viewMsg=f"Move gripper to marker {id}, then press q")

    ground_truth[id] = C.eval(ry.FS.position, ['l_gripper'])[0]

bot.home()
bot.wait(C)
del bot

print("Optimizing calibration")

def objective(q,t):
    transform = np.zeros((4,4))
    transform[:3, :3] = R.from_quat(q).as_matrix()
    transform[:3, 3] = t
    transform[3, 3] = 1.

    error = 0
    for q, measurement in measurements:
        C.setJointState(q)
        gripper_transform = np.linalg.inv(C.getFrame('l_gripper').getTransform())
        for id, z_cam in measurement.items():
            z_true = gripper_transform @ np.concatenate((ground_truth[id], [1]))
            error += np.linalg.norm((transform @ z_true)[:3] - z_cam)**2

    return error

def objective_grad(q,t):
    transform = np.zeros((4,4))
    transform[:3, :3] = R.from_quat(q).as_matrix()
    transform[:3, 3] = t
    transform[3, 3] = 1.

    grad_t = np.zeros(3)
    for q, measurement in measurements:
        C.setJointState(q)
        gripper_transform = np.linalg.inv(C.getFrame('l_gripper').getTransform())
        for id, z_cam in measurement.items():
            z_true = gripper_transform @ np.concatenate((ground_truth[id], [1]))
            grad_t = 2 * ((transform @ z_true)[:3] - z_cam)

    raise NotImplementedError("Gradient w.r.t. quaternion not implemented")


def constraint(q, t):
    return 1/2*np.linalg.norm(q)**2 - 1

def constraint_grad(q, t):
    grad = np.zeros(7)
    grad[3:] = q
    return grad

raise NotImplementedError("Optimization not implemented yet")