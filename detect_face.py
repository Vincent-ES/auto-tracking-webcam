import cv2 as cv
import mediapipe as mp

class FaceDetector:
    def __init__(self, model_selection = 1, min_detection_confidence = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection = model_selection,
            min_detection_confidence = min_detection_confidence
        )

    def find_face(self, frame):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.detector.process(frame_rgb)

        error_x = 0
        face_found = False

        if result.detections:
            face_found = True
            detection = result.detections[0]
            bbox = detection.location_data.relative_bounding_box

            face_center_x = bbox.xmin + (bbox.width / 2)
            error_x = face_center_x - 0.5

        return face_found, error_x