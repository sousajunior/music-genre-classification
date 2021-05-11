from load_data import load_data
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from plot_history import plot_history

DATASET_PATH = "extracted_data_with_10_segments_and_30_sec.json"


if __name__ == '__main__':
    inputs, targets = load_data(DATASET_PATH)

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

    # treina a rede
    history = model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_test, targets_test),
        epochs=150,
        batch_size=32  # (16 - 128)
    )

    # imprime um gráfico com o histórico do desempenho da rede
    plot_history(history)

    model.save("mlp_genre_classifier.h5")
