world: {}

### table

origin (world): { Q: [0, 0, .6], shape: marker, size: [.03] }
table (origin): { Q: [0, 0, -.05], shape: ssBox, size: [2.3, 1.24, .1, .02], color: [.3, .3, .3], contact, logical:{ } }

## two pandas
Prefix: "l_"
Include: <../../../../$RAI_PATH/panda/panda.g>
Prefix: False

## position them on the table
Edit l_panda_base (origin): { Q: "t(-.4 -.3 .0) d(90 0 0 1)", motors, joint: rigid }
#Edit l_panda_base (origin): { Q: "t(-.4008 -.2204 .0009) d(90 0 0 1)", motors, joint: rigid }

Edit l_panda_joint1_origin: { pose: [0, 0, 0.333, 1, 0, 0, 4.49499e-08] }
Edit l_panda_joint2_origin: { pose: [0, 0, 0, 0.707106, -0.707106, 0.000673419, 0.000673419] }
Edit l_panda_joint3_origin: { pose: [0, -0.316, 0, 0.707107, 0.707107, -0.000185814, 0.000185814] }
Edit l_panda_joint4_origin: { pose: [0.0825, 0, 0, 0.7071, 0.7071, 0.00304907, -0.00304907] }
Edit l_panda_joint5_origin: { pose: [-0.0825, 0.384, 0, 0.707097, -0.707097, 0.00371641, 0.00371641] }
Edit l_panda_joint6_origin: { pose: [0, 0, 0, 0.707105, 0.707105, 0.00177004, -0.00177004] }
Edit l_panda_joint7_origin: { pose: [0.088, 0, 0, 0.707107, 0.707107, 8.19741e-09, -8.19741e-09] }


## make gripper dofs inactive (unselected)
# Edit l_panda_finger_joint1: { joint_active: False }

### cameras

camera(world): {
 Q: "t(-0.01 -.2 2.) d(-150 1 0 0)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.5, 100]
}

l_cameraWrist(l_panda_joint7): {
 Q: [-0.0234024, 0.0460216, 0.166593, 0.393053, 0.00494562, -0.010527, -0.919442],
 shape: camera, size: [.1],
 focalLength: 0.495, width: 640, height: 360, zRange: [.1, 10]
}