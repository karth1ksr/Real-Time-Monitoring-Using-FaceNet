from PIL import Image
import numpy as np
import os
from threading import Thread
from numpy import asarray, expand_dims, load
import mtcnn
import cv2
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import pymysql
import cryptography
from keras_facenet import FaceNet
import datetime


class FaceRec:
    def __init__(self, camera_id, cam_no, cam_res):
        self.camera_id = camera_id
        self.detector = mtcnn.MTCNN()
        self.MyFaceNet = FaceNet()
        self.cam_no = cam_no
        self.cam_res = cam_res
        self.svm = None
        self.cursor = None
        self.counter = 0
        self.cnct = None
        self.face_confidence_threshold = 0.9

    def connect_database(self):
        self.cnct = pymysql.connect(
            user='enter_username',
            password='enter_password',
            host='enter_hostname',
            database='name_of_the_database_you_want_to_update'
        )
        self.cursor = self.cnct.cursor()

    def train_svm(self):

        data = load('trained_faces_enc.npz')
        trainX, trainy = data['arr_0'], data['arr_1']

        in_encoder = Normalizer(norm='l2')
        in_encoder.fit(trainX)
        trainX = in_encoder.transform(trainX)

        self.lb = LabelEncoder()
        self.lb.fit(trainy)

        trainy = self.lb.transform(trainy)

        self.svm = make_pipeline(MinMaxScaler(), SVC(kernel='rbf', C=1, gamma=0.01, probability=True))
        self.svm.fit(trainX, trainy)

    def process_frame(self, frame):
        image = Image.fromarray(frame)
        image = image.convert('RGB')
        pixels = asarray(image)

        results = self.detector.detect_faces(pixels)

        for result in results:
            if result['confidence'] >= self.face_confidence_threshold:
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                face = pixels[y1:y2, x1:x2]
                face = Image.fromarray(face)
                face = face.resize((160, 160))
                face = asarray(face)

                face = expand_dims(face, axis=0)
                signature = self.MyFaceNet.embeddings(face)
                signature = signature[0].reshape(1, -1)

                inp_encoder = Normalizer(norm='l2')
                inp_encoder.fit(signature)
                signature = inp_encoder.transform(signature)

                predicted_label = self.svm.predict(signature)
                yhat_prob = self.svm.predict_proba(signature)

                name_enc = predicted_label[0]
                probability = yhat_prob[0, name_enc] * 100

                outlb = self.lb.inverse_transform(predicted_label)
                my_name = outlb[0]

                if outlb and (probability > 70):
                    curr_t = datetime.datetime.now()
                    self.update_database(x1, y1, x2, y2, frame, my_name, curr_t)

                    label_text = f"{my_name} {probability:.2f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    label_width = text_size[0] + 10
                    label_height = text_size[1] + 5

                    cv2.rectangle(frame, (x1, y1 - label_height - 15), (x1 + label_width, y1 - 5), (0, 0, 255), -1)
                    cv2.putText(frame, label_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if self.counter == 0:
                        self.counter += 1

        if self.counter != 0:
            if self.counter == 1:
                self.retrieve_cropped_image(my_name)
            if 2 < self.counter < 6:
                self.display_cropped_image(frame)

            self.counter += 1

            if self.counter > 6:
                self.counter = 0

    def update_database(self, x1, y1, x2, y2, frame, pn, nt):
        cropped_face = frame[y1:y2, x1:x2]
        scaled_cropped_face = cv2.resize(cropped_face, (100, 100))
        _, cropcode = cv2.imencode('.jpg', scaled_cropped_face)
        binary_data = cropcode.tobytes()
        if self.camera_id == 0:
            query = "UPDATE pdata SET Cropped_img = %s, Update_time = %s, Cameraid = %s WHERE Name = %s"
            self.cursor.execute(query, (binary_data, nt, self.cam_no, pn, ))
            self.cnct.commit()
        else:
            if self.cam_res == 'in':
                query4 = "UPDATE pdata SET Cropped_img = %s, Update_time = %s,Status = 'In', Cameraid = %s WHERE Name = %s"
                self.cursor.execute(query4, (binary_data, nt, self.cam_no, pn,))
                self.cnct.commit()

            if self.cam_res == 'out':
                query5 = "UPDATE pdata SET Cropped_img = %s, Update_time = %s,Status = 'Out', Cameraid = %s WHERE Name = %s"
                self.cursor.execute(query5, (binary_data, nt, self.cam_no, pn,))
                self.cnct.commit()

    def retrieve_cropped_image(self, ab):

        query = "SELECT Cropped_img FROM pdata WHERE Name = %s"
        self.cursor.execute(query, (ab, ))
        op = self.cursor.fetchall()

        bimage = op[0][0]
        byte_d = bytes(bimage)
        npr = np.frombuffer(byte_d, np.uint8)
        self.cdbimage = cv2.imdecode(npr, cv2.COLOR_BGRA2BGR)

    def display_cropped_image(self, frame):

        x = frame.shape[1] - self.cdbimage.shape[1] - 10
        y = 10
        frame[y: y + self.cdbimage.shape[0], x: x + self.cdbimage.shape[1]] = self.cdbimage
        cv2.rectangle(frame, (x, y), (x + self.cdbimage.shape[1], y + self.cdbimage.shape[0]), (0, 255, 0), 1)

    def run(self):

        self.connect_database()
        self.train_svm()

        cap = cv2.VideoCapture(self.camera_id)

        while True:
            ret, frame = cap.read()

            self.process_frame(frame)

            cv2.imshow(f"Face Recognition App - Camera {self.cam_no}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


print("Do you want to access webcam?")
x = input("YES or NO :").lower()
if x == 'yes':
    face_recognition = FaceRec(0, 0, 'nil')
    face_recognition.run()
if x == 'no':
    nc = int(input("Enter the number of cameras you want to use:"))
    cameras = []
    for i in range(nc):
        cameras.append(FaceRec(input(str("Enter Camera address:")), int(input("Enter camera_index in number:")),
                                            input("Do you want the Camera to update In or Out:").lower()))

    threads = []
    for camera in cameras:
        thread = Thread(target=camera.run)
        threads.append(thread)

    for thread in threads:
        thread.start()
    for trh in threads:
        trh.join()  
