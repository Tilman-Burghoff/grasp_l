base { multibody, multibody_gravity: false }

l_floatX (base){ joint:transX, limits:[-2 2], mass:.01 }
l_floatY (l_floatX){ joint:transY, limits:[-2 2], mass:.01 }
l_floatZ (l_floatY){ joint:transZ, limits:[0 3], mass:.01, q: 1 }
l_floatBall (l_floatZ){ joint:quatBall, limits:[-1 -1 -1 -1 1 1 1 1], mass:.01 }

Prefix: "l_"
Include: <../panda/panda_gripper.g>
Prefix: False

l_gripper_base(l_floatBall): { Q:"t(-0.3 0 .1035) d(180 1 0 0) d(-90 0 0 1)", shape: marker, size: [.03] }
Edit l_panda_hand(l_gripper_base): {}

## define a gripper, palm and fingers

l_gripper(l_panda_hand): { Q: "d(180 0 1 0) d(90 0 0 1) t(0 0 -.1035)", shape: marker, size: [.03], color: [.9, .9, .9], logical: { is_gripper } }
l_palm(l_panda_hand): { Q: "d(90 1 0 0)", shape: capsule, color: [1.,1.,1.,.1], size: [.14, .07], contact: -3 }
#l_finger1(l_panda_finger_joint1): { Q: [0, 0.028, .035], shape: capsule, size: [.02, .03], color: [1., 1., 1., .2], contact: -2 }
#l_finger2(l_panda_finger_joint2): { Q: [0, -.028, .035], shape: capsule, size: [.02, .03], color: [1., 1., 1., .2], contact: -2 }
l_finger1(l_panda_finger_joint1): { Q: [0, 0.013 .03], shape: ssBox, size: [.02, .03, .05, .005], color: [1., 1., 1., .1], contact: -2 }
l_finger2(l_panda_finger_joint2): { Q: [0, -.013, .03], shape: ssBox, size: [.02, .03, .05, .005], color: [1., 1., 1., .1], contact: -2 }

l_dotA(l_panda_finger_joint1){ Q:[0, 0, .0451], shape:sphere, size:[.01], color:[1 0 0 .5] }
l_dotB(l_panda_finger_joint2){ Q:[0, 0, .0451], shape:sphere, size:[.01], color:[1 0 0 .5] }

Edit l_panda_finger_joint1: { q: .04 }

r_floatX (base){ joint:transX, limits:[-2 2], mass:.01 }
r_floatY (r_floatX){ joint:transY, limits:[-2 2], mass:.01 }
r_floatZ (r_floatY){ joint:transZ, limits:[0 3], mass:.01, q: 1 }
r_floatBall (r_floatZ){ joint:quatBall, limits:[-1 -1 -1 -1 1 1 1 1], mass:.01 }

Prefix: "r_"
Include: <../panda/panda_gripper.g>
Prefix: False

r_gripper_base(r_floatBall): { Q:"t(0.3 0 .1035) d(180 1 0 0) d(-90 0 0 1)", shape: marker, size: [.03] }
Edit r_panda_hand(r_gripper_base): {}

## define a gripper, palm and fingers

r_gripper(r_panda_hand): { Q: "d(180 0 1 0) d(90 0 0 1) t(0 0 -.1035)", shape: marker, size: [.03], color: [.9, .9, .9], logical: { is_gripper } }
r_palm(r_panda_hand): { Q: "d(90 1 0 0)", shape: capsule, color: [1.,1.,1.,.1], size: [.14, .07], contact: -3 }
#r_finger1(r_panda_finger_joint1): { Q: [0, 0.028, .035], shape: capsule, size: [.02, .03], color: [1., 1., 1., .2], contact: -2 }
#r_finger2(r_panda_finger_joint2): { Q: [0, -.028, .035], shape: capsule, size: [.02, .03], color: [1., 1., 1., .2], contact: -2 }
r_finger1(r_panda_finger_joint1): { Q: [0, 0.013 .03], shape: ssBox, size: [.02, .03, .05, .005], color: [1., 1., 1., .1], contact: -2 }
r_finger2(r_panda_finger_joint2): { Q: [0, -.013, .03], shape: ssBox, size: [.02, .03, .05, .005], color: [1., 1., 1., .1], contact: -2 }

r_dotA(r_panda_finger_joint1){ Q:[0, 0, .0451], shape:sphere, size:[.01], color:[1 0 0 .5] }
r_dotB(r_panda_finger_joint2){ Q:[0, 0, .0451], shape:sphere, size:[.01], color:[1 0 0 .5] }

Edit r_panda_finger_joint1: { q: .04 }

## Cameras ##
l_cameraWrist(l_palm): {
 Q: "d(90 0 1 0) t(0.0 0.05 0.05)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.1, 10]
}

l_panda_collCameraWrist(l_cameraWrist): {
 Q: "d(90 0 1 0) t(-.02 0 0)"
 , shape: capsule, color: [1.,1.,1.,.2], size: [.05, .03], contact: -3
}

r_cameraWrist(r_palm): {
 Q: "d(90 0 1 0) t(0.0 0.05 0.05)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.1, 10]
}

r_panda_collCameraWrist(r_cameraWrist): {
 Q: "d(90 0 1 0) t(-.02 0 0)"
 , shape: capsule, color: [1.,1.,1.,.2], size: [.05, .03], contact: -3
}
