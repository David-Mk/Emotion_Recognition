import csv
import cv2
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split as tts


# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Detection models
mp_holistic = mp.solutions.holistic

# Capture video feed. In case of errors, try swap number inside (camera index)
capture = cv2.VideoCapture(0)


def capture():

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

                # LATER!!! ADD CODE TO CHECK IF COORDS.CSV FILE EXISTS AND SKIP MANUAL UNCOMMENTING PART
                
                # Code bellow shall be used when you already have coords.csv file generated and coordinates inside are set properly.
                # Try extract landmarks, pass if not detected
                try:

                    # Change this parameter to adjust mood (Happy, Sad, Jambo, Rock, Sleepy, Surprised, Angry)
                    cond = 'Sleepy'

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
                    row.insert(0, cond)

                    # Write coordinates to .csv file
                    with open('coords.csv', mode='a', newline='') as f:
                        writer = csv.writer(
                            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(row)

                except:
                    pass

                # In case  coords.csv file is generated for the first time, uncomment code bellow to create coordinate grid set
                #
                # coords = len(result.pose_landmarks.landmark) + len(result.face_landmarks.landmark)
                # landmarks = ['Class']
                #
                # for i in range(1, coords + 1):
                #
                #    landmarks += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i), 'v{}'.format(i)]
                #
                # with open('coords.csv', 'w', newline = '') as f:
                #
                #    writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                #    writer.writerow(landmarks)

                # Show feed and detector
                cv2.imshow('Video feed', image)
                cv2.waitKey(10)

        except KeyboardInterrupt:
            pass

capture()