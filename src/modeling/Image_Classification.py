import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# %matplotlib inline
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


def plot_sample(index):
    plt.figure(figsize=(10, 1))
    plt.imshow(X_train[index])


plot_sample(10)
y_train[0:5]

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

X_train_scaled.shape

y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)
y_train_categorical[:5]

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(3000, activation="relu"),
        keras.layers.Dense(3000, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid"),
    ]
)


model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train_scaled, y_train_categorical, epochs=50)

classes[np.argmax(model.predict(X_test_scaled)[10])]
classes[y_test[10][0]]


def plot_sample_test(index):
    plt.figure(figsize=(10, 1))
    plt.imshow(X_test[index])


plot_sample_test(10)
