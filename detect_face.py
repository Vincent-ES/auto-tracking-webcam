import cv2 as cv

cap = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('auto-tracking-webcam/haar_face.xml')


while True:
    isTrue, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_blur = cv.GaussianBlur(gray, (7,7), 0)

    if not isTrue:
        break

    faces_rect = haar_cascade.detectMultiScale(gray_blur, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), thickness=2)

    cv.imshow('webcam', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()