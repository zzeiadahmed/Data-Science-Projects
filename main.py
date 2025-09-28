import cv2
import time
import winsound  # plays alert sound on Windows

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open webcam
cap = cv2.VideoCapture(0)

# Frame-based counters
closed_eyes_frames = 0
drowsy_threshold = 30  # number of frames eyes can be closed before alert

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            closed_eyes_frames += 1
        else:
            closed_eyes_frames = 0

        # Draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Trigger alarm
        if closed_eyes_frames > drowsy_threshold:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            winsound.Beep(1000, 1000)  # Beep sound

    cv2.imshow('Driver Drowsiness Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

