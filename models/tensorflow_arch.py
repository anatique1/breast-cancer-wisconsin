# app/models/tensorflow_arch.py

from tensorflow import keras
from tensorflow.keras import layers

def build_tabular_model(input_dim=30):
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),  # primera capa Input
        layers.Dense(64, activation="relu"), #  más capacidad (64)
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # 1 neurona, Sigmoid
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy", # pérdida binaria
                  metrics=["accuracy"])
    return model

def build_image_model(num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=(32,32,3)),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_audio_model(num_classes=35):
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(None, None, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
