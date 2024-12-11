import cv2

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained data (ensure the path is correct)
recognizer.read('TrainingImageLabel/trainer.yml')

# Load the face detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # If confidence is less than 100, means "0" is perfect match
        if confidence < 100:
            id = f"Person {id}"
            confidence = f"  {round(100 - confidence)}%"
        else:
            id = "unknown"
            confidence = f"  {round(100 - confidence)}%"

        cv2.putText(img, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
