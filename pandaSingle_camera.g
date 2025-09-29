world: {}

### table

origin (world): { Q: [0, 0, .6], shape: marker, size: [.03] }
table (origin): { Q: [0, 0, -.05], shape: ssBox, size: [2.3, 1.24, .1, .02], color: [.3, .3, .3], contact, logical:{ } }

## two pandas
Prefix: "l_"
Include: <../panda/panda.g>
Prefix: False

## position them on the table
# Edit l_panda_base (origin): { Q: "t(-.4 -.3 .0) d(90 0 0 1)", motors, joint: rigid }
Edit l_panda_base (origin): { Q: "t(-.4008 -.2204 .0009) d(90 0 0 1)", motors, joint: rigid }

## make gripper dofs inactive (unselected)
# Edit l_panda_finger_joint1: { joint_active: False }

### cameras

camera(world): {
 Q: "t(-0.01 -.2 2.) d(-150 1 0 0)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.5, 100]
}

l_cameraWrist_oo(l_gripper): {
 Q: [ 0.03056593, -0.0509864, 0.04855423, 0.02046703, 0.0075157, 0.99956733, -0.01974268],
 shape: camera, size: [.1],
 focalLength: 0.495, width: 640, height: 360, zRange: [.1, 10]
}

l_cameraWrist_o(l_cameraWrist_oo): {
 Q: "t(0.0325 .0 0.)",
 shape: camera, size: [.1],
 focalLength: 0.495, width: 640, height: 360, zRange: [.01, 10]
}

l_cameraWrist(l_panda_joint7): {
 Q: [-0.0245442, 0.0477194, 0.16876, 0.393552, 0.00961641, -0.00293342, -0.919247],
 shape: camera, size: [.1],
 focalLength: 0.495, width: 640, height: 360, zRange: [.1, 10]
}