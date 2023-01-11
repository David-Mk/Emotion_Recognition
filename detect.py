from chardet import detect
import cv2
import numpy as np
import mediapipe as mp
import pickle as pl
import pandas as pd

# Importing trained model (RandomForest in this case)
with open('emotions.pkl', 'rb') as f:

    model = pl.load(f)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Detection models
mp_holistic = mp.solutions.holistic

# Capture video feed. In case of errors, try swap number inside (camera index)
capture = cv2.VideoCapture(0)


def detect():

    # Capture video feed and handling keyboard interrupt
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:

        try:
            while capture.isOpened():

                # Writing feed data
                r, frame = capture.read()

                # Process the video feed. Recolouring and adding result
                frame = cv2.flip(frame, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                result = hol.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw detection on feed
                # Face landmarks
                mp_draw.draw_landmarks(
                    image, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    mp_draw.DrawingSpec(color=(80, 110, 10),
                                        thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

                # Body landmarks
                mp_draw.draw_landmarks(
                    image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(245, 117, 66),
                                        thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Right hand landmarks
                mp_draw.draw_landmarks(
                    image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(80, 22, 10),
                                        thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

                # Left hand landmarks
                mp_draw.draw_landmarks(
                    image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(80, 22, 10),
                                        thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

                # Try extract landmarks, pass if not detected
                try:

                    # Extract pose landmarks
                    pose = result.pose_landmarks.landmark
                    current_pose = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Extract face landmarks
                    face = result.face_landmarks.landmark
                    current_face = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                    # Concatenate face and body
                    row = current_pose + current_face

                    # Make detections
                    X = pd.DataFrame([row])
                    detection_class = model.predict(X)[0]
                    detection_prob = model.predict_proba(X)[0]

                    # Grab coordinates
                    coords = tuple(np.multiply(np.array(
                        (result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                         result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x)),
                        [640, 480]).astype(int))

                    # Rendering rectangle and text
                    #cv2.rectangle(image, (coords[0], coords[1]+5), (coords[0]+len(detection_class)*20, coords[1]-30), (245, 117, 16), -1)
                    #cv2.putText(image, detection_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display configurations
                    # Rendering rectangle
                    cv2.rectangle(image, (0, 0), (250, 60),
                                  color=(39, 39, 186), thickness=-1)

                    # Display current detections
                    cv2.putText(image, 'DETECT', (95, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
                    cv2.putText(image, detection_class.split(' ')[
                                0], (95, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)

                    # Display probability
                    cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (20, 20, 20), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(detection_prob[np.argmax(detection_prob)], 2)), (
                        10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)

                except:
                    pass

                # Show feed and detector
                cv2.imshow('Video feed', image)
                cv2.waitKey(10)

        except KeyboardInterrupt:
            pass

detect()