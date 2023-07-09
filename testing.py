from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from numpy import load

data = load('trained_faces_enc.npz')
X_train, y_train = data['arr_0'], data['arr_1']


data1 = load('test_faces_enc.npz')
encodings, labels = data1['arr_0'], data1['arr_1']

in_encoder = Normalizer(norm='l2')
in_encoder.fit(X_train)
trainX = in_encoder.transform(X_train)

svm = make_pipeline(MinMaxScaler(), SVC(kernel='rbf', C=1, gamma=0.01, probability=True))
svm.fit(X_train, y_train)

op = svm.predict(encodings)
print("Accuracy:", accuracy_score(labels, op) * 100)
print("Recall:", recall_score(labels, op, average='weighted') * 100)
print("Precision:", precision_score(labels, op,  average='weighted') * 100)
print("F1 Score:", f1_score(labels, op, average='weighted') * 100)
print("Confusion Matrix:")
print(confusion_matrix(labels, op))
