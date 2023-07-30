#import necessary modules
import numpy as np
from numpy import load
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load('trained_faces_enc.npz')
encodings, labels = data['arr_0'], data['arr_1']

n_folds = 10
kf = StratifiedKFold(n_splits=n_folds, shuffle=True)

accuracies = []
precisions = []
recalls = []
f1_scores = []

for train_index, val_index in kf.split(encodings, labels):

    X_train, X_val = encodings[train_index], encodings[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    in_encoder = Normalizer(norm='l2')
    in_encoder.fit(X_train)
    trainX = in_encoder.transform(X_train)
    valX = in_encoder.transform(X_val)

    lb = LabelEncoder()
    lb.fit(y_train)

    y_train = lb.transform(y_train)
    y_val = lb.transform(y_val)

    svm = make_pipeline(MinMaxScaler(), SVC(kernel='rbf', C=1, gamma=0.01, probability=True))
    svm.fit(trainX, y_train)

    y_pred = svm.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

print("Average Accuracy:", avg_accuracy*100)
print("Average Precision:", avg_precision*100)
print("Average Recall:", avg_recall*100)
print("Average F1 Score:", avg_f1_score*100)
