from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
import cv2
import numpy as np
from mediapipe import solutions

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
def dive(gestureResult):
    return 1

def draw_gesture_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    gestures = detection_result.gestures
    annotated_image = np.copy(rgb_image)
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        gesture = gestures[idx][0].category_name
        handedness_name = handedness_list[idx][0].category_name

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image,f"{handedness_name}:"+f"{gesture}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

if __name__ == '__main__':
    # STEP 2: Create an HandLandmarker object.
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
        running_mode=VisionRunningMode.VIDEO,min_hand_presence_confidence=0.5,num_hands=2)
    with GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(0)
        timestamp = 0
        l = 0
        while True:
            ret, frame = cap.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            gesture_recognition_result = recognizer.recognize_for_video(mp_image, int(timestamp))
            timestamp += 1.0
            annotated_image = draw_gesture_on_image(frame, gesture_recognition_result)
            cv2.imshow("Test", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break