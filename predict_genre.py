import os
from settings import NUMBER_OF_MFCCS, NUMBER_OF_SEGMENTS, TRACK_DURATION
from consolemenu.console_menu import clear_terminal
import numpy as np
from song_features_extractor import extract_mfcc_from_one_song
from tensorflow.keras.models import load_model
from consolemenu import ConsoleMenu, SelectionMenu
from consolemenu.items import MenuItem
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

    # limpa os warnings que aparecem relacionados a GPU
    clear_terminal()

    X = extract_mfcc_from_one_song(
        music_file_path,
        num_mfcc=NUMBER_OF_MFCCS,
        num_segments=NUMBER_OF_SEGMENTS,
        track_duration=TRACK_DURATION
    )


    # não adiciona a terceira dimensão no array quando a rede é do tipo RNN
    if network_type != 'rnn': X = X[..., np.newaxis]
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
    while True:
        menu_title = "Classificador automático de gêneros músicais"
        menu_subtitle = "Criado por: Carlinhos de Sousa Junior"
        network_types = ["mlp", "cnn", "rnn"]
        # obtém uma lista com todas as músicas dentro do diretório /songs
        musics = os.listdir('songs')

        network_selection_menu = SelectionMenu(
            network_types,
            title=menu_title,
            subtitle=menu_subtitle,
            prologue_text="Para iniciar escolha uma das opções de redes neurais do menu abaixo:"
        )

        # exibe o menu para seleção do algoritmo a ser utilizado
        network_selection_menu.show()

        music_selection_menu = SelectionMenu(
            musics,
            title=menu_title,
            subtitle=menu_subtitle,
            prologue_text="Certo, agora você precisa escolher uma das músicas abaixo para que o classificador tente adivinhar qual o gênero dela:",
        )

        # exibe o menu para seleção da música a ser analisada
        music_selection_menu.show()

        selected_network = network_types[network_selection_menu.selected_option]
        selected_music = musics[music_selection_menu.selected_option]

        # faz a previsão dos gêneros que possuem semelhança com a música selecionada
        predicted_genres = predict_genre(
            f"songs/{selected_music}",
            network_type=selected_network
        )

        result_menu = ConsoleMenu(
            title=menu_title,
            subtitle=menu_subtitle,
            prologue_text="De acordo com o algoritmo de '{}', a música '{}' tem similaridade com os seguintes gêneros:".format(
                selected_network, selected_music)
        )

        # cria os itens do menu de resultados obtidos pela previsão
        for predicted_genre in predicted_genres:
            result_menu.append_item(MenuItem("Gênero: {}, Precisão: {:.2f}%".format(
                predicted_genre['label'], predicted_genre['accuracy'])))

        result_menu.show()
