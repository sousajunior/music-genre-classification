import numpy as np


def predict_genre(model, X, y):
    """Prevê o gênero
    :param model (Sequential): Modelo da rede que realizará a previsão
    :param X (ndarray): Features da música
    :param y (int32): Gênero da música
    """

    X = X[np.newaxis, ...]
    prediction = model.predict(X)

    predicted_index = np.argmax(prediction, axis=1)

    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))
