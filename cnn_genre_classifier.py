import numpy as np
from load_data import load_data
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from plot_history import plot_history
from predict_genre import predict_genre

DATASET_PATH = "extracted_data_with_10_segments_and_30_sec.json"


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

    # adiciona um novo eixo aos dados
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Gera um modelo de rede CNN
    :param input_shape (tuple): Dados de entrada da rede
    :return model: Modelo da CNN
    """

    # cria a topologia da rede
    model = keras.Sequential()

    # 1ª camada de convolução
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2ª camada de convolução
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3ª camada de convolução
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # nivela a saída e a coloca em uma camada densa
    model.add(keras.layers.Flatten())
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
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compila a rede
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # imprime um resumo com as camadas do modelo de rede criada
    model.summary()

    # treina a rede
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        batch_size=128, # era 32, mudei pra 128 para testar
        epochs=150
    )

    # imprime um gráfico com o histórico do desempenho da rede
    plot_history(history)

    # avalia o desempenho da rede com os dados de teste
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("A acurácia no teste é de: {}".format(test_accuracy))
    print("O erro no teste é de: {}".format(test_error))

    model.save("cnn_genre_classifier.h5")

    # TODO: precisa ajustar o mecanismo de predição...
    # - Pegar uma música, extrair os dados dela, obter o gênero e enviar pra predição
    # X = X_test[100]
    # y = y_test[100]
    # predict_genre(model, X, y)
