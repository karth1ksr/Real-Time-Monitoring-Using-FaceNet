##Face Recognition using Multi-CNN

Face Recognition with Facenet.
Config files for my GitHub profile.
Hello, Here I am going to share with you the face recognition model that I have created with the help of MTCNN, FaceNet, and SVM.
First of all, the basics of the face recognition system are detecting and extracting the face. This is done by Multi-Task CNN, it detects the face based on the features.
After detecting the face, it extracts and passes it to the FaceNet model to find the embeddings. These embeddings are in the form of vectors.
The embeddings produced by FaceNet are passed to the SVM classifier for classification. ( Note: The SVM classifier is pre-trained with the training dataset ).

The recognized face will be cropped and will be updated in the database. I will also share code for connecting to the database and creating a table.


The provided code is created for the purpose of detecting faces in the CCTV camera and updating the values in the database. If you want to try this code from your webcam and do not want to update any values in the database, make sure to checkout this python file which the executes the face recognition by accessing your webcam and without updating any database values

Note:
The face encoding code is not written by me and it was taken from the given website, https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
So, I am grateful for the person to share the code.
