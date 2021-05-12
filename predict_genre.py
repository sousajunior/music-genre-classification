import os
import numpy as np
from extract_mfcc_of_one_music import extract_mfcc_of_one_music
from tensorflow.keras.models import load_model
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

genres = {0: 'blues', 1: 'clássico', 2: 'country', 3: 'discoteca',
          4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}


def predict_genre(music_file_path, network_type='mlp'):
    """Prevê o gênero de uma determinada música utilizando o modelo de rede escolhido
    :param music_file_path (str): Caminho da música a ser realizada a predição
    :param network_type (str): Tipo da rede a ser utilizada na predição (mlp, cnn, rnn)
    :return predicted_genres_ranking (dict): Ranking com os três melhores gêneros que a rede identificou para a música selecionada
    """

    # carrega a rede com o model type correspondente
    model = load_model(f'{network_type}_genre_classifier.h5')
    X = extract_mfcc_of_one_music(music_file_path)

    X = X[np.newaxis, ...]
    prediction = model.predict(X)

    ranking_size = 3
    # ranking com as posições das melhores predições, ordenadas da melhor para a pior
    ranking = (-prediction).argsort()[0][:ranking_size]
    predicted_genres_ranking = [dict() for x in range(ranking_size)]

    # preenche um dictionary com o tamanho do ranking
    for index in range(ranking_size):
        # preenche o dictionary com a % de precisão na chave "accuracy"
        predicted_genres_ranking[index]['accuracy'] = prediction[0][ranking[index]]
        # preenche o dictionary com o nome do gênero na chave "label"
        predicted_genres_ranking[index]['label'] = genres[ranking[index]]

    return predicted_genres_ranking


if __name__ == '__main__':
    while(True):
        music_file_path = input(
            "Insira o caminho da música que você quer descobrir o gênero:")

        network_type = input(
            "Qual o tipo de RNA que você deseja usar ? (mlp, cnn, rnn):")

        print('Certo, bora analisar sua música então...')

        predicted_genres = predict_genre(
            music_file_path, network_type=network_type)

        print(
            f"Os três gêneros que mais se identificam com {music_file_path} são: ")

        for index, predicted_genre in enumerate(predicted_genres, start=1):
            print("[{}] - Gênero: {}, Precisão: {:.2f}%".format(
                index, predicted_genre['label'], predicted_genre['accuracy']))
