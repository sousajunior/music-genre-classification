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


# mlp_genre_classifier.h5 - acc: 0.59%, erro: 1.88%
if __name__ == '__main__':
    # obtém os dados de treinamento, validação e teste
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        test_size=0.25, validation_size=0.2)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # construindo a arquitetura da rede neural
    model = keras.Sequential([
        # camada de entrada
        keras.layers.Flatten(input_shape=input_shape),

        # 1ª camada oculta
        keras.layers.Dense(512, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2ª camada oculta
        keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3ª camada oculta
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # camada de saída
        keras.layers.Dense(10, activation="softmax")
    ])

    # compilar a rede
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # imprime um resumo da rede criada
    model.summary()

    # treina a rede
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=150,
        batch_size=32  # (16 - 128)
    )

    # imprime um gráfico com o histórico do desempenho da rede
    plot_history(history)

    # avalia o desempenho da rede com os dados de teste
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("A taxa de precisão no teste é de: {:.2f}%".format(test_accuracy))
    print("O taxa de erro no teste é de: {:.2f}%".format(test_error))

    model.save("mlp_genre_classifier.h5")
