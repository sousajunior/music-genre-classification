import json
import os
import math
import librosa

# directory with all musics tracks
DATASET_PATH = "genres"

# default sample rate of GTZAN tracks
SAMPLE_RATE = 22050

# Testar:
#   Alterar o num_segments para 10 ou algo > que 5


def create_mfcc_features_file(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5, track_duration=30):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Caminho onde se encontram os dados
        :param num_mfcc (int): Número de coeficientes MFCCs para extrair
        :param n_fft (int): Intervalo a considerar para aplicar a FFT. Medido em # de amostras
        :param hop_length (int): Janela deslizante para FFT. Medido em # de amostras
        :param: num_segments (int): Número de segmentos no qual cada amostra da música será dividida
        :param: track_duration (int): Número de duração da música em segundos
        """

    print("Extração das características mfcc inicializada...")

    # estrutura de dados para guardar o mapping, labels e MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_track = SAMPLE_RATE * track_duration
    samples_per_segment = int(samples_per_track / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(
        samples_per_segment / hop_length)

    # itera sobre todas as sub pastas de gênero
    for label_index, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):

        # garante que estamos processando uma sub pasta no nível de gênero
        if dirpath is not dataset_path:

            # genre\blues => ["genre", "blues"]
            # obtém o nome do genêro pelo caminho do arquivo
            # está utilizando o padrão de caminhos do windows
            semantic_label = dirpath.split('\\')[-1]

            # salva o nome do gênero na estrutura de mapping
            data["mapping"].append(semantic_label)
            print("\nProcessando: {}".format(semantic_label))

            # processa todos os arquivos de audio no diretório de gênero
            for filename in filenames:

                # carrega o arquivo de audio
                file_path = os.path.join(dirpath, filename)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # processa todos os segmentos dos arquivos de audio
                for segment_index in range(num_segments):

                    # calcula o inicio e o fim da amostra do segmento atual
                    start_sample_index = samples_per_segment * segment_index
                    finish_sample_index = start_sample_index + samples_per_segment

                    # extrai o mfcc
                    mfcc = librosa.feature.mfcc(
                        signal[start_sample_index:finish_sample_index],
                        sample_rate,
                        n_mfcc=num_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length
                    )
                    mfcc = mfcc.T

                    # guarda apenas as características do mfcc que possui o numero esperado de vetores,
                    # pois se um mfcc não possuir a quantidade correta esperada, ele pode estragar os dados existentes
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(label_index-1)
                        print("{}, segmento:{}".format(
                            file_path, segment_index+1))

    # nome do arquivo a ser criado para guardar as características extraídas
    json_path = f"extracted_data_with_{num_segments}_segments_and_{track_duration}_sec.json"

    print("\nSalvando os dados como: {}".format(json_path))

    # salva os MFCCs em um arquivo .json
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print("Extração das características mfcc finalizada!")


if __name__ == "__main__":
    # Descomentar as chamadas abaixo para criar todos os arquivos necessários para o projeto

    # create_mfcc_features_file(DATASET_PATH, num_segments=5, track_duration=10)
    # create_mfcc_features_file(DATASET_PATH, num_segments=5, track_duration=30)
    # create_mfcc_features_file(DATASET_PATH, num_segments=10, track_duration=10)
    # create_mfcc_features_file(DATASET_PATH, num_segments=10, track_duration=30)
    # create_mfcc_features_file(DATASET_PATH, num_segments=15, track_duration=10)
    # create_mfcc_features_file(DATASET_PATH, num_segments=15, track_duration=30)

    # Comentar a chamada abaixo caso descomente as chamadas acima
    create_mfcc_features_file(DATASET_PATH, num_segments=10, track_duration=30)
