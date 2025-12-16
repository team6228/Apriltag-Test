import cv2
import numpy as np
from pupil_apriltags import Detector

# [TODO]
# Look more to calibrating stuff
# Fix focal lengths optical center
# Fix april tag size
# Add config.py
# MOST IMPRTANT
# Add networktables and test the latency

cap = cv2.VideoCapture(0)

fx, fy = 600, 600
cx, cy = 320, 240
camera_params = (fx, fy, cx, cy)

at_detector = Detector(families='tag36h11')

tag_size = 0.1524

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size
    )

    for tag in tags:
        if tag.tag_id == 1:
            tvec = tag.pose_t.flatten()
            rvec = tag.pose_R

            yaw = np.arctan2(rvec[1, 0], rvec[0, 0]) * 180 / np.pi
            pitch = np.arctan2(
                -rvec[2, 0],
                np.sqrt(rvec[2, 1]**2 + rvec[2, 2]**2)
            ) * 180 / np.pi
            roll = np.arctan2(rvec[2, 1], rvec[2, 2]) * 180 / np.pi

            print(f"Tag ID: {tag.tag_id}")
            print(f"Position (meters): X={tvec[0]:.3f}, Y={tvec[1]:.3f}, Z={tvec[2]:.3f}")
            print(f"Angles (degrees): Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")

            for i in range(4):
                p1 = tuple(tag.corners[i].astype(int))
                p2 = tuple(tag.corners[(i + 1) % 4].astype(int))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("AprilTag Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
