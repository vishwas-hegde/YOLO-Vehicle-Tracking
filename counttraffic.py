import time
from collections import defaultdict

import cv2
import numpy as np
import itertools
from ultralytics import YOLO

def segment_crosswalk(frame):
    """Segment the crosswalk from the frame.
    Credits: https://stackoverflow.com/questions/71989250/how-could-i-improve-my-opencv-program-to-detect-only-the-crosswalk
    """
    # define the range of brown color in BGR (the values were obtained through trial and error)
    lower_r = 144
    lower_g = 142
    lower_b = 140
    upper_r = 162
    upper_g = 159
    upper_b = 157
    # define area range for the contour
    area_min = 73
    area_max = 156

    # define range of color in BGR
    lower_brown = np.array([lower_b, lower_g, lower_r])
    upper_brown = np.array([upper_b, upper_g, upper_r])

    thresh = cv2.inRange(frame, lower_brown, upper_brown)

    # apply morphology close to fill interior regions in mask
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # get contours
    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # filter on area
    good_contours = []
    centroids = []
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > area_min and area < area_max:
            good_contours.append(c)
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

    # get the contours that are close to each other
    grouped_contours = []
    for i, j in itertools.combinations(range(len(good_contours)), 2):
        dist = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
        if dist < 98:
            grouped_contours.append(good_contours[i])
            grouped_contours.append(good_contours[j])

    # get the hull and centroid
    if grouped_contours:
        # combine good contours
        contours_combined = np.vstack(grouped_contours)
        # get convex hull
        hull = cv2.convexHull(contours_combined)
        M = cv2.moments(hull)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return hull, [cx, cy]

def put_text_with_bg(image, text, position, font, font_scale, font_color, bg_color, thickness):
    """Insert text with a background rectangle on an image."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, cv2.FILLED)
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "TrafficVideo.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# store cars in a dict
track_history = {}
count_cars = 0
count_bikes = 0
count_humans = 0

segment = True

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        if segment:
            hull, centroid = segment_crosswalk(frame)
            line_x, line_y = centroid
            segment = False

        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], (0, 255, 255))

        # Blend the overlay with the original image
        alpha = 0.2  # Transparency factor (0 = transparent, 1 = opaque)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = frame.copy()
        # cv2.line(annotated_frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)
        # cv2.line(annotated_frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 2)

        for box, track_id, cls in zip(boxes, track_ids, classes):
            x, y, w, h = map(int, box)
            label = f"ID: {track_id}"

            classlabel = f"{model.names[cls]}"

            # Draw the bounding box and label
            cv2.rectangle(annotated_frame, (x-int(w/2), y-int(h/2)), (x+int(w/2), y+int(h/2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, classlabel, (x-int(w/2), y-int(h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(annotated_frame, label, (x-int(w/2), y-int(h/2) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if track_id not in track_history and (cls == 2 or cls == 1 or cls == 0):
                # add the new track_id to the track_history
                track_history[track_id] = {}
                x_center = x   # x_center
                y_center = y   # y_center
                track_history[track_id]["cross"] = 0
                track_history[track_id]["x_center"] = x_center
                track_history[track_id]["y_center"] = y_center
                track_history[track_id]["class"] = cls

                if cls == 1 or cls == 2:
                    # consider the center in x-axis for cars and bikes
                    if x_center < line_x:
                        track_history[track_id]["status"] = 1   # status 1 means the object is on the left side of the line
                    elif x_center > line_x:
                        track_history[track_id]["status"] = 2   # status 2 means the object is on the right side of the line
                elif cls == 0:
                    # consider the center in y-axis for humans
                    # as the height of the bounding box is more for humans, consider center at h/4
                    temp_y_center = y_center + h/4
                    time.sleep(0.5)
                    if temp_y_center < line_y:
                        track_history[track_id]["status"] = 1    # status 1 means the object is on the top side of the line
                    elif temp_y_center > line_y:
                        track_history[track_id]["status"] = 2    # status 2 means the object is on the bottom side of the line
            elif track_id in track_history:
                # update the x_center and y_center for the track_id
                x_center = x
                y_center = y
                if cls == 0:
                    # as the height of the bounding box is more for humans, consider center at h/4
                    y_center = y + h/4
                track_history[track_id]["x_center"] = x_center
                track_history[track_id]["y_center"] = y_center

        for track_id, object in track_history.items():
            # check if the object has crossed the line
            if object["class"] == 1 or object["class"] == 2:
                # if the x_center shifts beyond the centroid of the segment then the object has crossed the line
                if object["status"] == 1 and object["x_center"] > line_x and object["cross"] == 0:
                    object["cross"] = 1     # set the cross flag to 1 to indicate the object has crossed the line
                    if object["class"] == 1:
                        count_bikes += 1
                    elif object["class"] == 2:
                        count_cars += 1
                if object["status"] == 2 and object["x_center"] < line_x and object["cross"] == 0:
                    object["cross"] = 1
                    if object["class"] == 1:
                        count_bikes += 1
                    elif object["class"] == 2:
                        count_cars += 1
            elif object["class"] == 0:
                # if the y_center shifts beyond the centroid of the segment then the object has crossed the line
                if object["status"] == 1 and object["y_center"] > line_y and object["cross"] == 0:
                    object["cross"] = 1
                    count_humans += 1
                if object["status"] == 2 and object["y_center"] < line_y and object["cross"] == 0:
                    object["cross"] = 1
                    count_humans += 1

        # print(track_history)
        # cv2.putText(annotated_frame, f"Count: {count_cars} \t Bicycle: {count_bikes}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # display the count of cars and bikes and humans on the frame
        put_text_with_bg(annotated_frame, f"Car Count: {count_cars}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),(255, 0, 0), 2)
        put_text_with_bg(annotated_frame, f"Bike Count: {count_bikes}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (255, 0, 0),2)
        put_text_with_bg(annotated_frame, f"Human Count: {count_humans}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (255, 0, 0),2)


        # resize the annotated frame to 90% of the original size to fit the screen
        annotated_frame = cv2.resize(annotated_frame, (int(frame_width * 0.9), int(frame_height * 0.9)))

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()