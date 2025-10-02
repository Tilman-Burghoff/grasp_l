from robot_api import RobotAPI
from vision import get_point_clouds
from scenes import get_config_table
from motions import move_to_drop, grasp_motion_global, look_with_angle
from grasping import contact_graspnet_inference
import komo_paths as kp
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
import torch

MARKER_POS = np.array([-.5, .3, .65])
NUMBER_OF_TRIES = 3 # number of tries to find a feasible grasp
ON_REAL = True

def find_grasps(C, robot_api: RobotAPI, disturb=False) -> tuple[bool, np.ndarray, np.ndarray]:
    if disturb:
        q = C.getJointState()
        q += np.random.randn(len(q)) * .1
        robot_api.moveTo(q)
        C.setJointState(q)
    pcs, rgbs = get_point_clouds(C, camera_frame_names, robot_api, on_real=ON_REAL, verbose=0)#, distance_boundaries=(0.15, 0.7))


    torch.cuda.empty_cache()
    grasps, scores = contact_graspnet_inference(pcs[0], rgbs[0], local_regions=False, filter_grasps=False, forward_passes=2, verbose=1, from_top=10)


    if len(grasps) == 0:
        return False, None, None


    camera_frame = C.getFrame('l_cameraWrist')
    grasp_camera_frame = C.addFrame('grasp_camera', 'l_cameraWrist')
    grasp_frame = C.addFrame('grasp').setShape(ry.ST.marker, [.2])
    C.addFrame('approach', 'grasp')

    filtered_grasps = []
    for i, g in enumerate(grasps):
        grasp_camera_frame.setRelativePose(g)
        grasp_global = grasp_camera_frame.getPose()
        if grasp_global[2] < .65:
            continue
        filtered_grasps.append((grasp_global, scores[i]))
    print(len(filtered_grasps), "grasps after filtering")

    if len(filtered_grasps) == 0:
        return False, None, None

    filtered_grasps.sort(key = lambda g: g[1])#np.linalg.norm(g[:3] - MARKER_POS)) 



    komo = ry.KOMO(C, 1,1,10,True)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', 'grasp'], ry.OT.eq)

    q = C.getJointState()
    for g in filtered_grasps:
        C.setJointState(qHome)
        grasp_frame.setPose(g[0])
        komo.updateRootObjects(C)
        ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
        C.view()
        komo.view(True, f'candidate grasp, {ret.feasible=}')
        if ret.feasible:
            print("Found a feasible grasp")
            try:
                C.setJointState(qHome)
                approach, grasp = grasp_motion_global(C, grasp_frame.getPose())
                break
            except:
                continue
    else: # we move to this branch if no break occurred
        C.setJointState(q)
        return False, None, None
    print("Grasp Found")
    C.setJointState(q)
    return True, approach, grasp

    

camera_frame_names = ["l_cameraWrist"]
robot_api = RobotAPI(verbose=2, use_foundation_stereo=False, address="tcp://130.149.82.15:1234", on_real=ON_REAL) if ON_REAL else None

C = get_config_table(verbose=1)
qHome = C.getJointState()
C.addFrame('marker').setPosition(MARKER_POS).setShape(ry.ST.marker, [.2])
path = look_with_angle(C, 'marker', distance=.4, angle=np.pi/6, verbose=1)

if ON_REAL:
    robot_api.move(path, [3.])
C.setJointState(path[-1])

for i in range(NUMBER_OF_TRIES):
    success, approach, grasp = find_grasps(C, robot_api, disturb=(i>0))
    if success:
        break
else:
    raise RuntimeError("No feasible grasps found")
    

if ON_REAL:
    robot_api.home()
    robot_api.move(approach, [5.])
    robot_api.move(grasp, [5.])
    robot_api.gripper_close()
    robot_api.home()
    path = move_to_drop(C, 'marker', distance=.1, verbose=1)
    robot_api.move(path, [5.])
    robot_api.gripper_open()
    robot_api.home()
    # robot_api.close()
