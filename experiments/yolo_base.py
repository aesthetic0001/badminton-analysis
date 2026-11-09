from ultralytics import YOLO
import cv2

file_name = "./shot1.png"

object_detection_model = YOLO("yolo26m.pt")
pose_detection_model = YOLO("yolo26m-pose.pt")

objects = object_detection_model(file_name)[0]
poses = pose_detection_model(file_name)[0]

print(objects.boxes)
print(poses.keypoints)

frame = objects.plot()
frame = poses.plot(img=frame, boxes=False)

# Display combined results
cv2.imshow("Combined Detections", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save combined frame
cv2.imwrite("combined.jpg", frame)