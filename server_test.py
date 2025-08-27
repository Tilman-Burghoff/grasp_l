from robot_api import RobotAPI
from vision import get_point_clouds
from scenes import get_config_table
from motions import move_to_look, grasp_motion
from grasping import contact_graspnet_inference

ON_REAL = True
camera_frames = ["l_cameraWrist", "r_cameraWrist"]
robot_api = RobotAPI(verbose=2, use_foundation_stereo=False, address="tcp://130.149.82.15:1234", on_real=ON_REAL) if ON_REAL else None
robot_api.gripper_close()
