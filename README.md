Face Recognition with Facenet.
Config files for my GitHub profile.
Hello, Here I am going to share with you the face recognition model which I have created with the help of MTCNN, FaceNet, and SVM.
First of all, the basics of the face recognition system are detecting and extracting the face. This is done by Multi-Task CNN, it detects the face based on the features.
After detecting the face, it extracts and passes it to the FaceNet model to find the embeddings. These embeddings are in the form of vectors.
The embeddings produced by the FaceNet are passed to the SVM classifier for classification. ( Note: The SVM classifier is pre-trained with the testing dataset ).

The recognized face will be cropped and will be updated in the database. I will also share code for connecting to the database and creating a table.
