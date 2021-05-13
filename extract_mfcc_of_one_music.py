import math
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# directory with all musics tracks
DATASET_PATH = "genres"

# default sample rate of GTZAN tracks
SAMPLE_RATE = 22050

def extract_mfcc_of_one_music(music_file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10, track_duration=30):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param music_file_path (str): Caminho onde se encontra a música
        :param num_mfcc (int): Número de coeficientes MFCCs para extrair
        :param n_fft (int): Intervalo a considerar para aplicar a FFT. Medido em # de amostras
        :param hop_length (int): Janela deslizante para FFT. Medido em # de amostras
        :param: num_segments (int): Número de segmentos no qual cada amostra da música será dividida
        :param: track_duration (int): Número de duração da música em segundos
        :return inputs (ndarray): Conjunto de dados com as features extraídas da música
        """

    print("Extraindo as características da música...")

    # estrutura de dados para guardar o mapping, labels e MFCCs
    data = {
        "mfcc": []
    }

    samples_per_track = SAMPLE_RATE * track_duration
    samples_per_segment = int(samples_per_track / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(
        samples_per_segment / hop_length)

    # carrega o arquivo de audio
    signal, sample_rate = librosa.load(music_file_path, sr=SAMPLE_RATE)

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

    # faz umas transformações para ajustar os dados
    inputs = np.array(data['mfcc'])

    print("Características da música extraídas com sucesso!")

    return inputs[num_segments-1]
