import numpy as np
from load_data import load_data
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from plot_history import plot_history

DATASET_PATH = "extracted_data_with_10_segments_and_30_sec.json"


def prepare_datasets(test_perc_size=None, validation_perc_size=None):
    X, y = load_data(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_perc_size)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_perc_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    
    predicted_index = np.argmax(prediction, axis=1)

    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


if __name__ == '__main__':
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        test_perc_size=0.25, validation_perc_size=0.2)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=150)

    plot_history(history)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("A acurácia no teste é de: {}".format(test_accuracy))

    model.save("cnn_genre_classifier.h5")

    # X = X_test[100]
    # y = y_test[100]
    # predict(model, X, y)

