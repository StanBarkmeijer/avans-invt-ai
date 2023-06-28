from keras.models import load_model
from PIL import Image
import numpy as np
import cv2

files = [
    "num_0.png",
    "num_1.png",
    # "num_2.png",
    "num_3.png",
    "num_4.png",
    "num_5.png",
    "num_6.png",
    "num_7.png",
    "num_8.png",
    "num_9.png"
]

values = [
    0,
    1,
    6,
    4,
    8,
    3,
    2,
    3,
    9
]

models = [
    "mnist.h5",
    "mnist2.h5",
    "src/mnist_model_1.h5",
    "src/mnist_model_2.h5",
    "src/mnist_model_3.h5"
]

actual_models = []

for i in range(len(models)):
    actual_models.append(load_model(models[i]))

def predict_digit(img):
    results = []

    for i in range(len(actual_models)):
        res = actual_models[i].predict(img)[0]
        results.append(np.argmax(res))

    return results

for i in range(len(files)):
    accuracy = 0

    image = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32') / 255

    results = predict_digit(image)
    print("File: " + files[i])
    print("Actual: " + str(values[i]))

    print("")
    
    for j in range(len(results)):
        print("Model " + str(models[j]) + ": " + str(results[j]))
        if results[j] == values[i]:
            accuracy += 1

    print("Accuracy: " + str((accuracy / len(results) * 100)) + "%")
    print("")