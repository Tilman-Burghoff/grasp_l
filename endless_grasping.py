from robot_api import RobotAPI
from vision import get_point_clouds
from scenes import get_config_table
from motions import grasp_motion_global_from_look, look_with_angle, move_to_place
from grasping import contact_graspnet_inference
import komo_paths as kp
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
import torch

MARKER_POS = np.array([-.5, .3, .65])
NUMBER_OF_TRIES = 10 # number of tries to find a feasible grasp

def find_grasps(C, robot_api: RobotAPI, disturb=False) -> tuple[bool, np.ndarray, np.ndarray]:
    if disturb:
        q = C.getJointState()
        q += np.random.randn(len(q)) * .01
        robot_api.moveTo(q)
        C.setJointState(q)
    pcs, rgbs = get_point_clouds(C, camera_frame_names, robot_api, on_real=True, verbose=0)#, distance_boundaries=(0.15, 0.7))


    torch.cuda.empty_cache()
    grasps, scores = contact_graspnet_inference(pcs[0], rgbs[0], local_regions=False, filter_grasps=False, forward_passes=2, verbose=0, from_top=10)


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
        if grasp_global[2] < .65 or grasp_global[0] > 0.3:
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
        C.setJointState(q)  
        grasp_frame.setPose(g[0])
        komo.updateRootObjects(C)
        ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
        #C.view()
        #komo.view(True, f'candidate grasp, {ret.feasible=}')
        if ret.feasible:
            print("Found a feasible grasp")
            try:
                C.setJointState(q)
                approach, grasp = grasp_motion_global_from_look(C, grasp_frame.getPose(), verbose=10)
                print("Grasp executable")
                C.setJointState(q)
                return True, approach, grasp
            except:
                continue

    C.setJointState(q)
    return False, None, None

    
print(ry.raiPath(''))
camera_frame_names = ["l_cameraWrist"]
robot_api = RobotAPI(verbose=1, use_foundation_stereo=False, address="tcp://130.149.82.15:1234", on_real=True)

C = ry.Config()
C.addFile("pandaSingle_camera.g")
print("Config loaded")
qHome = C.getJointState()
marker = C.addFrame('marker').setPosition(MARKER_POS).setShape(ry.ST.marker, [.2])

while True:
    path = look_with_angle(C, 'marker', distance=.4, angle=np.pi/6, verbose=0)

    robot_api.move(path, [5.])
    C.setJointState(path[-1])

    for i in range(NUMBER_OF_TRIES):
        success, approach, grasp = find_grasps(C, robot_api, disturb=(i>0))
        if success:
            break
    else:
        raise RuntimeError("No feasible grasps found")
        

    robot_api.moveAutoTimed(approach, 0.5, 0.5)
    robot_api.move(grasp, [5.])
    C.setJointState(grasp[-1])
    gripper_pose = C.getFrame('l_gripper').getPose()
    robot_api.gripper_close()
    robot_api.home()
    place = move_to_place(C, gripper_pose)
    robot_api.moveTo(place)
    robot_api.gripper_open()
    C.setJointState(place)
    dropoff_point = C.getFrame('l_gripper').getPosition()
    dropoff_point[2] = 0.65
    marker.setPosition(dropoff_point)

