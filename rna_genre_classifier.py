import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_SET_PATH = "extracted_data_with_10_segments_and_30_sec.json"


def load_data(dataset_path):
    print(f"Carregando o arquivo {dataset_path}")

    # abre o arquivo com os dados
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

        # converte as listas em um array numpy
        inputs = np.array(data['mfcc'])
        targets = np.array(data['labels'])

        print("Arquivo carregado com sucesso!")

        return inputs, targets


def plot_history(history):
    _, axs = plt.subplots(2)

    # cria o subplot da acurácia
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # cria o subplot do erro
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == '__main__':
    inputs, targets = load_data(DATA_SET_PATH)

    # separando os dados para testar e treinar
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs,
        targets,
        test_size=0.3
    )

    # construindo a arquitetura da rede neural
    model = keras.Sequential([
        # camada de entrada
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

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

    # treinar a rede
    history = model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_test, targets_test),
        epochs=150,
        batch_size=32  # (16 - 128)
    )

    # imprime a acurácia e o erro em cada época
    plot_history(history)
