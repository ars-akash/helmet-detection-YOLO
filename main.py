import cv2
import numpy as np


weights = 'helmet-detection-yolov3/yolov3-helmet.weights'
cfg = 'helmet-detection-yolov3/yolov3-helmet.cfg'

# Load fine-tuned YOLOv3 weights and configuration files
net = cv2.dnn.readNet(weights, cfg)

# Load class labels
classes = ['helmet']

# Generate a random set of colors to represent each class label
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set the minimum confidence threshold for object detection
conf_threshold = 0.5

# Set the non-maximum suppression threshold for object detection
nms_threshold = 0.4

file_path = 'C:/Users/divya/Downloads/videoplayback.mp4'
# Open a video stream
cap = cv2.VideoCapture(file_path)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Create a blob from the input frame and pass it through the YOLOv3 network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to store the detected object bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the detected objects and filter out weak detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] == 'helmet':
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw the filtered bounding boxes and labels on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            left, top, width, height = boxes[i]
            label = '{}: {:.2f}'.format(classes[class_ids[i]], confidences[i])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with the bounding boxes and labels
    cv2.imshow('Helmet Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close
