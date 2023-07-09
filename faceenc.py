from PIL import Image
from os import listdir
from os.path import isdir
from mtcnn.mtcnn import MTCNN
from numpy import load, asarray, expand_dims, savez_compressed
from keras_facenet import FaceNet


def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def load_faces(directory):
    faces = []
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces


def load_dataset(directory):
    X, y = [], []
    for subdir in listdir(directory):
        path = directory + subdir + '/'

        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


trainX, trainy = load_dataset('Test Image/')
savez_compressed('test_faces.npz', trainX, trainy)


def get_embedding(model, face_pixels):
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]


data = load('test_faces.npz')
trainX, trainy = data['arr_0'], data['arr_1']
print('Loaded: ', trainX.shape, trainy.shape)
MyFaceNet = FaceNet()

xtrain_enc = []
for face_pixels in trainX:
    embedding = get_embedding(MyFaceNet, face_pixels)
    xtrain_enc.append(embedding)
xtrain_enc = asarray(xtrain_enc)
print(xtrain_enc.shape)
savez_compressed('test_faces_enc.npz', xtrain_enc, trainy)

