import robotic as ry
import pyrealsense2 as rs
import numpy as np
import cv2

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    print("Device:", dev.get_info(rs.camera_info.name))
    print("Serial Number:", dev.get_info(rs.camera_info.serial_number))
exit()

from vision import foundation_stereo_call

# Configure the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB and Infrared streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # IR Left

# Start the pipeline
pipeline.start(config)

C = ry.Config()
C.view()

try:
    while True:
        # Wait for a coherent set of frames
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame(2)  # 1 for IR Left, 2 for IR Right if available

        if not color_frame or not ir_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)


        # Display images
        cv2.imshow('RGB Image', color_image)
        cv2.imshow('IR Image', ir_image)

        key = cv2.waitKey(1)
        if key == 27:  # Esc to exit
            break
        
        pc, rgb = foundation_stereo_call(color_image, ir_image)
        pc_frame = C.addFrame(f"pointCloud")
        pc_frame.setPointCloud(pc, rgb)
        C.view(False)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
