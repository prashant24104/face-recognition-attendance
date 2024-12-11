import cv2
import os
import numpy as np
from PIL import Image

# Initialize the recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    # Loop through all the image paths and load the Ids and images
    for imagePath in imagePaths:
        # Print the current image path
        print(f"Processing image: {imagePath}")

        # Load the image and convert it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Convert the PIL image to a numpy array
        imageNp = np.array(pilImage, 'uint8')

        # Get the Id from the image filename
        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
        except ValueError:
            print(f"Skipping file {imagePath}: invalid filename format.")
            continue

        # Extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        
        if len(faces) == 0:
            print(f"No faces found in image: {imagePath}")
        
        # If a face is found, append it to the list along with the Id
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)

    print(f"Total faces detected: {len(faceSamples)}")
    print(f"Total IDs collected: {len(Ids)}")
    return faceSamples, Ids

# Main execution
faces, Ids = getImagesAndLabels('TrainingImage')
if len(faces) == 0 or len(Ids) == 0:
    print("Error: No training data found. Please check the image directory and format.")
else:
    recognizer.train(faces, np.array(Ids))
    recognizer.save('TrainingImageLabel/trainer.yml')
    print("Training complete. Model saved.")
