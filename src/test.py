from keras.models import load_model
from PIL import Image
import numpy as np

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
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    results = []

    for i in range(len(actual_models)):
        res = actual_models[i].predict(img)[0]
        results.append(np.argmax(res))

    return results

for i in range(len(files)):
    img = Image.open(files[i])
    results = predict_digit(img)
    print("File: " + files[i])
    print("Actual: " + str(values[i]))
    
    for j in range(len(results)):
        print("Model " + str(models[j]) + ": " + str(results[j]))

    print("")