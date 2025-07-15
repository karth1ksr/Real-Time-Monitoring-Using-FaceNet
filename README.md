# 🧠 Real-Time Face Recognition System using FaceNet

## 📌 Project Objective

This project implements a real-time **face recognition-based monitoring system** designed for organizational surveillance — such as in schools, campuses, and offices. It detects and recognizes individuals from a webcam/video feed and logs their presence into a database.

### 🧩 Key Features
- Real-time face detection with **MTCNN (P-Net, R-Net, O-Net)**
- Face embedding with **FaceNet** (128-D vectors)
- Classification using **Support Vector Machine (SVM)**
- Auto-updates recognition logs into a connected database
- Modular codebase for both DB-connected and standalone systems

---

## 🛠️ Technologies Used

- Python
- TensorFlow/Keras
- OpenCV
- MTCNN
- Scikit-learn
- PymySQL

---

## 🗃️ Project Structure

Face-Recognition-with-Facenet/
│
├── create_db.py # Initializes SQLite database
├── cross_validation.py # Evaluates SVM classifier performance
├── faceenc.py # Generates face embeddings from training images
├── mfsvm.py # Main application with DB integration
├── mfsvm_no_db.py # Same as above but without database logging
├── testing.py # Standalone script for testing recognition accuracy
└── README.md # You're reading it!


---

## 🎯 How It Works

1. **Face Detection**  
   Uses **MTCNN** to detect and crop faces in real-time from a webcam feed.

2. **Embedding Generation**  
   Each detected face is converted into a 128-dimensional vector using **FaceNet**.

3. **Face Classification**  
   The vector is passed to an **SVM classifier** which predicts the person's identity.

4. **Database Logging**  
   Upon recognition, the system logs user details into a connected **SQLite database**.

---

## 📷 Sample Use Case

✅ Campus monitoring system  
✅ Office attendance system  
✅ Secure access control using face recognition  
✅ Scalable for new faces (simply add images and rerun `faceenc.py`)

---

## 📌 Setup Instructions

```bash
# Step 1: Create the database
python create_db.py

# Step 2: Generate face embeddings from labeled images
python faceenc.py

# Step 3: Train and test the model
python cross_validation.py
python testing.py

# Step 4: Run the real-time system
python mfsvm.py           # With DB logging
python mfsvm_no_db.py     # Without DB
```
## Future Improvements
1. Add GUI using Streamlit or Flask

2. Enable face registration through webcam

3. Extend to recognize unknown faces (open-set recognition)

4. Deploy on cloud with public API for face verification

## Acknowledgements

Face Encoding Code from machinelearnningmastery[https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/]
MTCNN Implementation[https://github.com/ipazc/mtcnn]

Pretrained Keras FaceNet Model by nyoki-mtl[https://github.com/nyoki-mtl/keras-facenet]

## LICENSE 
This project is licensed under te MIT License.
Feel free to use, modify, and distribute it with attribution.
