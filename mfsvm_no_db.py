# importing necessary packages.
from PIL import Image
import numpy as np
import os
from numpy import asarray, expand_dims, load
import mtcnn
import cv2
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder
from keras_facenet import FaceNet

#Creating class for cleaner code
class SVMTrainer:
    # This class contains the functions of training svm classifier
    def __init__(self):
        self.MyFaceNet = FaceNet()
        self.svm = None
        self.lb = None

    def train(self):
        # Loading the pre-trained encodings of the training images
        data = load('trained_faces_enc.npz')
        trainX, trainy = data['arr_0'], data['arr_1']

        # Normalizing the encodings to reduce noise
        in_encoder = Normalizer(norm='l2')
        in_encoder.fit(trainX)
        trainX = in_encoder.transform(trainX)

        # Encoding the persons name into numerical value to find the probability
        self.lb = LabelEncoder()
        self.lb.fit(trainy)
        trainy = self.lb.transform(trainy)

        # Training part of the SVM Classifier
        self.svm = make_pipeline(SVC(kernel='rbf', C=1, gamma=0.01, probability=True))
        self.svm.fit(trainX, trainy)

class FaceRec:
    # This contains necessary functions to detect faces, extract embeddings, and predict the person
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.detector = mtcnn.MTCNN()
        self.trainer = SVMTrainer()
        self.face_confidence_threshold = 0.9

    def process_frame(self, frame):
        # Converting the feed from the camera to an image
        image = Image.fromarray(frame)
        image = image.convert('RGB')
        pixels = asarray(image)

        # Detecting the faces in the frame
        results = self.detector.detect_faces(pixels)

        for result in results:
            if result['confidence'] >= self.face_confidence_threshold:
                # Putting boxes for the detected faces in the frame
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                # Extracting the faces from the frame
                face = pixels[y1:y2, x1:x2]
                face = Image.fromarray(face)
                face = face.resize((160, 160))
                face = asarray(face)

                # Preprocessing the input for MyFaceNet
                face = expand_dims(face, axis=0)
                signature = self.trainer.MyFaceNet.embeddings(face)
                signature = signature[0].reshape(1, -1)

                # Predict the label and probability using SVM classifier
                predicted_label = self.trainer.svm.predict(signature)
                yhat_prob = self.trainer.svm.predict_proba(signature)

                name_enc = predicted_label[0]
                probability = yhat_prob[0, name_enc] * 100

                outlb = self.trainer.lb.inverse_transform(predicted_label)
                my_name = outlb[0]

                # This executes the below part only if the probability of the recognized face is above 70 to avoid misclassification
                if outlb and (probability > 70):
                    # Display the box and name along with the probability for the recognized face
                    label_text = f"{my_name} {probability:.2f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    label_width = text_size[0] + 10
                    label_height = text_size[1] + 5

                    cv2.rectangle(frame, (x1, y1 - label_height - 15), (x1 + label_width, y1 - 5), (0, 0, 255), -1)
                    cv2.putText(frame, label_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        # Train the SVM classifier
        self.trainer.train()

        # Capture feed from the camera
        cap = cv2.VideoCapture(self.camera_id)

        while True:
            ret, frame = cap.read()

            # Calling the process_frame function
            self.process_frame(frame)

            # Display the frame with face recognition results
            cv2.imshow("Face Recognition", frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close the windows
        cap.release()
        cv2.destroyAllWindows()

# This part asks the user to check with webcam or not
print("Do you want to access webcam?")
x = input("YES or NO: ").lower()
if x == 'yes':
    # Uses Webcam
    # calling the FaceRecognition class with camera_id 0 for default webcam
    face_recognition = FaceRec(0)
    face_recognition.run()
else:
    print("The process cannot be executed!!")

