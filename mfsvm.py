# importing necessary packages.
from PIL import Image
import numpy as np
import os
from numpy import asarray, expand_dims, load
import mtcnn
import cv2
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler
import pymysql
from keras_facenet import FaceNet
import datetime
from threading import Thread

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
        self.svm = make_pipeline(MinMaxScaler(), SVC(kernel='rbf', C=1, gamma=0.01, probability=True))
        self.svm.fit(trainX, trainy)


class FaceRec:
    # This contains necessary funtions to detect faces, extract embeddings, predicting the person and updating the db
    def __init__(self, camera_id, cam_no, cam_res):
        self.camera_id = camera_id
        self.detector = mtcnn.MTCNN()
        self.trainer = SVMTrainer()
        self.cam_no = cam_no
        self.cam_res = cam_res
        self.cursor = None
        self.counter = 0
        self.cnct = None
        self.face_confidence_threshold = 0.9

    def connect_database(self):
        # This part establishes connection to the database
        self.cnct = pymysql.connect(
            user='enter_username',
            password='enter_password',
            host='provide_the_hostname',
            database='enter_db_name'
        )
        self.cursor = self.cnct.cursor()

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

                # Normalize the face embedding for noise reduction
                inp_encoder = Normalizer(norm='l2')
                inp_encoder.fit(signature)
                signature = inp_encoder.transform(signature)

                # Predict the label and probability using SVM classifier
                predicted_label = self.trainer.svm.predict(signature)
                yhat_prob = self.trainer.svm.predict_proba(signature)

                name_enc = predicted_label[0]
                probability = yhat_prob[0, name_enc] * 100

                outlb = self.trainer.lb.inverse_transform(predicted_label)
                my_name = outlb[0]

                # This executes the below part only if the probability of the recognized face is above 70 to avoid misclassification
                if outlb and (probability > 70):

                    curr_t = datetime.datetime.now()
                    updater = DatabaseUpdater(self.cursor, self.cnct, self.cam_no, self.camera_id, self.cam_res)
                    updater.update(x1, y1, x2, y2, frame, my_name, curr_t)
                    # Display the box and name along with the probability for the recognized face
                    label_text = f"{my_name} {probability:.2f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    label_width = text_size[0] + 10
                    label_height = text_size[1] + 5

                    cv2.rectangle(frame, (x1, y1 - label_height - 15), (x1 + label_width, y1 - 5), (0, 0, 255), -1)
                    cv2.putText(frame, label_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if self.counter == 0:
                        self.counter += 1
        # The counter is used to display the cropped image on the frame only for certain time limit
        if self.counter != 0:
            if self.counter == 1:
                self.retrieve_cropped_image(my_name)
            if 2 < self.counter < 6:
                self.display_cropped_image(frame)

            self.counter += 1

            if self.counter > 6:
                self.counter = 0

    def retrieve_cropped_image(self, ab):
        # Retrieve the cropped image from the database
        query = "SELECT Cropped_img FROM pdata WHERE Name = %s"
        self.cursor.execute(query, (ab,))
        op = self.cursor.fetchall()

        # Convert the binary data of the image into an image
        bimage = op[0][0]
        byte_d = bytes(bimage)
        npr = np.frombuffer(byte_d, np.uint8)
        self.cdbimage = cv2.imdecode(npr, cv2.COLOR_BGRA2BGR)

    def display_cropped_image(self, frame):

        # Display the retrieved cropped image on the frame
        x = frame.shape[1] - self.cdbimage.shape[1] - 10
        y = 10
        frame[y: y + self.cdbimage.shape[0], x: x + self.cdbimage.shape[1]] = self.cdbimage
        cv2.rectangle(frame, (x, y), (x + self.cdbimage.shape[1], y + self.cdbimage.shape[0]), (0, 255, 0), 1)

    def run(self):

        # Connect to the database and train the SVM classifier
        self.connect_database()
        self.trainer.train()

        # Capture feed from the camera
        cap = cv2.VideoCapture(self.camera_id)

        while True:
            ret, frame = cap.read()

            # Calling the process_frame function
            self.process_frame(frame)

            # Display the frame with face recognition results
            cv2.imshow(f"Face Recognition App - Camera {self.cam_no}", frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close the windows
        cap.release()
        cv2.destroyAllWindows()


class DatabaseUpdater:
    # This class is used to update the database based on the results from process_frame

    def __init__(self, cursor, cnct, cam_no, camera_id, cam_res):
        self.cursor = cursor
        self.cnct = cnct
        self.cam_no = cam_no
        self.camera_id = camera_id
        self.cam_res = cam_res

    def update(self, x1, y1, x2, y2, frame, pn, nt):

        # Crop the face region from the frame, resize it and convert it into binary data to store in db
        cropped_face = frame[y1:y2, x1:x2]
        scaled_cropped_face = cv2.resize(cropped_face, (100, 100))
        _, cropcode = cv2.imencode('.jpg', scaled_cropped_face)
        binary_data = cropcode.tobytes()

        if self.camera_id == 0:

            # Update the database with the cropped face and other details
            query = "UPDATE pdata SET Cropped_img = %s, Update_time = %s, Cameraid = %s WHERE Name = %s"
            self.cursor.execute(query, (binary_data, nt, self.cam_no, pn))
            self.cnct.commit()
        else:
            if self.cam_res == 'in':

                # Update the database with the cropped face and other details
                query4 = "UPDATE pdata SET Cropped_img = %s, Update_time = %s, Status = 'In', Cameraid = %s WHERE Name = %s"
                self.cursor.execute(query4, (binary_data, nt, self.cam_no, pn))
                self.cnct.commit()

            if self.cam_res == 'out':

                # Update the database with the cropped face and other details
                query5 = "UPDATE pdata SET Cropped_img = %s, Update_time = %s, Status = 'Out', Cameraid = %s WHERE Name = %s"
                self.cursor.execute(query5, (binary_data, nt, self.cam_no, pn))
                self.cnct.commit()


# This part asks the user to check with webcam or not
print("Do you want to access webcam?")
x = input("YES or NO: ").lower()
if x == 'yes':
    # Uses Webcam
    # calling the FaceRecognition class
    face_recognition = FaceRec(0, 0, 'nil')
    face_recognition.run()
if x == 'no':
    # Uses cameras from the external source

    nc = int(input("Enter the number of cameras you want to use:"))
    cameras = []
    # Getting the details of the cameras to capture the feed
    for i in range(nc):
        camera_id = input(str("Enter Camera address:"))
        cam_no = int(input("Enter camera_index in number:"))
        cam_res = input("Do you want the Camera to update In or Out:").lower()
        cameras.append(FaceRec(camera_id, cam_no, cam_res))

    threads = []
    # Performing thread to run the multiple feeds simultaneously
    for camera in cameras:
        thread = Thread(target=camera.run)
        threads.append(thread)

    for thread in threads:
        thread.start()
    for trh in threads:
        trh.join()

