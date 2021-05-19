import numpy as np
from load_data import load_data
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from plot_history import plot_history
from settings import DATASET_PATH

def prepare_datasets(test_size=None, validation_size=None):
    """Carrega os dados e os divide em treinamento, validação e teste.
    :param test_size (float): Valor entre [0, 1] indicando a porcentagem de dados que devem ser separados para teste
    :param validation_size (float): Valore entre [0, 1] indicando a porcentagem de dados que devem ser separados para validação
    :return X_train (ndarray): Conjunto de dados de treinamento
    :return X_validation (ndarray): Conjunto de dados de validação
    :return X_test (ndarray): Conjunto de dados de teste
    :return y_train (ndarray): Conjunto de dados de objetivo do treinamento
    :return y_validation (ndarray): Conjunto de dados de objetivo da validação
    :return y_test (ndarray): Conjunto de dados de objetivo do teste
    """

    # carrega o arquivo com as features extraídas das músicas
    X, y = load_data(DATASET_PATH)

    # separa os dados em treinamento, teste e validação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# rnn_genre_classifier.h5 - acc: 0.77%, erro: 0.80%
# def build_model(input_shape):
#     """Gera um modelo de rede RNN-LSTM
#     :param input_shape (tuple): Dados de entrada da rede
#     :return model: Modelo da RNN
#     """

#     # cria a topologia da rede
#     model = keras.Sequential()

#     # 1ª camada LSTM
#     model.add(keras.layers.LSTM(
#         64, input_shape=input_shape, return_sequences=True))

#     # 2ª camada LSTM
#     model.add(keras.layers.LSTM(64))

#     # camada densa + dropout pra evitar o overfitting
#     model.add(keras.layers.Dense(64, activation='relu'))
#     model.add(keras.layers.Dropout(0.3))

#     # camada de saída
#     model.add(keras.layers.Dense(10, activation='softmax'))

#     return model

# rnn_genre_classifier_1.h5 - acc: 0.81%, erro: 0.63%
# def build_model(input_shape):
#     """Gera um modelo de rede RNN-LSTM
#     :param input_shape (tuple): Dados de entrada da rede
#     :return model: Modelo da RNN
#     """

#     # cria a topologia da rede
#     model = keras.Sequential()

#     # 1ª camada LSTM
#     model.add(keras.layers.LSTM(
#         64, input_shape=input_shape, return_sequences=True))
#     model.add(keras.layers.Dropout(0.3))

#     # 2ª camada LSTM
#     model.add(keras.layers.LSTM(128))
#     model.add(keras.layers.Dropout(0.3))

#     # camada densa + dropout pra evitar o overfitting
#     model.add(keras.layers.Dense(64, activation='relu'))
#     model.add(keras.layers.Dropout(0.3))

#     # camada de saída
#     model.add(keras.layers.Dense(10, activation='softmax'))

#     return model

# rnn_genre_classifier_2.h5 - acc: 0.89%, erro: 0.48%
def build_model(input_shape):
    """Gera um modelo de rede RNN-LSTM
    :param input_shape (tuple): Dados de entrada da rede
    :return model: Modelo da RNN
    """

    # cria a topologia da rede
    model = keras.Sequential()

    # 1ª camada LSTM
    model.add(keras.layers.LSTM(
        64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))

    # 2ª camada LSTM
    model.add(keras.layers.LSTM(128, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.LSTM(256))
    model.add(keras.layers.Dropout(0.3))

    # camada densa + dropout pra evitar o overfitting
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # camada de saída
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == '__main__':
    # obtém os dados de treinamento, validação e teste
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        test_size=0.25, validation_size=0.2)

    # cria a rede
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # compila a rede
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # imprime um resumo com as camadas do modelo de rede criada
    model.summary()

    # treina a rede
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        batch_size=128,  # era 32, mudei pra 128 para testar
        epochs=150
    )

    # imprime um gráfico com o histórico do desempenho da rede
    plot_history(history)

    # avalia o desempenho da rede com os dados de teste
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("A taxa de precisão no teste é de: {:.2f}%".format(test_accuracy))
    print("O taxa de erro no teste é de: {:.2f}%".format(test_error))

    # salva o modelo da rede treinada em um arquivo
    model.save("rnn_genre_classifier.h5")
