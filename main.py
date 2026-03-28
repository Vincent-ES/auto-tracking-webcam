import cv2 as cv
from detect_face import FaceDetector

def main():
    detector = FaceDetector()
    cap = cv.VideoCapture('Test.mp4')

    fps = 0
    while cap.isOpened():
        isTrue, frame = cap.read()
        if not isTrue:
            break

        if(not fps%3):
            found, error_x = detector.find_face(frame)

        if found:
            print(f"error_x: {error_x:.2f}")

            h, w, _ = frame.shape
            cv.line(frame, (int(w/2), 0), (int(w/2), h), (255, 0, 0), 1) # 畫面中心線
            cv.putText(frame, f"Error: {error_x:.2f}", (50, 50), 
            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv.imshow('Controller View', frame)
        fps += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()