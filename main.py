from robot_api import RobotAPI
from vision import get_point_clouds
from scenes import get_config_table
from motions import move_to_look, grasp_motion
from grasping import contact_graspnet_inference
import komo_paths as kp

import robotic as ry

print(ry.raiPath(''))

ON_REAL = True
camera_frame_names = ["l_cameraWrist"]
robot_api = RobotAPI(verbose=2, use_foundation_stereo=False, address="tcp://130.149.82.15:1234", on_real=ON_REAL) if ON_REAL else None

C = get_config_table(verbose=1)

C.addFrame('marker').setPosition([-.5, .3, .65]).setShape(ry.ST.marker, [.2])
path,_ = kp.look_at_target(C, 'marker', .4, verbose=True, show_result=True)

if ON_REAL:
    robot_api.move(path, [3.])
C.setJointState(path[-1])

pcs, rgbs = get_point_clouds(C, camera_frame_names, robot_api, on_real=ON_REAL, verbose=0)

import torch
torch.cuda.empty_cache()
grasps = contact_graspnet_inference(pcs[0], rgbs[0], local_regions=False, filter_grasps=False, forward_passes=2, verbose=1, from_top=10)

grasp_camera_frame = C.addFrame('grasp_camera', 'camera')
grasp_frame = C.addFrame('grasp').setShape(ry.ST.marker, [.2])


komo = ry.KOMO(C, 1,1,10,True)
komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
komo.addObjective([], ry.FS.poseDiff, ['l_gripper', 'grasp'], ry.OT.eq)

for g in grasps:
   # grasp_camera_frame.setRelativePose(g)
    grasp_frame.setPose(g) #grasp_camera_frame.getPose())
    komo.updateRootObjects(C)
    ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
    C.view()
    komo.view(True, f'candidate grasp, {ret.feasible=}')
    if ret.feasible:
        print("Found a feasible grasp")
        break

komo.view(True, "grasp pose")



try:
    approach, retract = grasp_motion(C, grasp_pose, verbose=1, arm_prefix="l_")
except:
    approach, retract = grasp_motion(C, grasp_pose, verbose=1, arm_prefix="r_")

if ON_REAL:
    robot_api.move(approach, [10.])
    C.view(True)
    robot_api.gripper_close()
    robot_api.move(retract, [10.])

    # robot_api.close()
