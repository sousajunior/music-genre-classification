# TODO: Revisar essas configurações no final

# diretório com todas as músicas do GTZAN
GTZAN_DIRECTORY = "genres"

# directory with all musics tracks to train
TEST_MUSICS_DIRECTORY = "test"

# taxa de amostragem padrão das músicas da base GTZAN
SAMPLE_RATE = 22050

# número de segmentos utilizado
NUMBER_OF_SEGMENTS = 15

# número de MFCCs utilizados
NUMBER_OF_MFCCS = 13

# número de MFCCs utilizados
TRACK_DURATION = 30

# nome do arquivo que contém os dados (features) das músicas
TRAIN_DATASET_PATH = f"extracted_data_with_{NUMBER_OF_SEGMENTS}_segments_and_{TRACK_DURATION}_sec.json"

# número a ser usado no "train_test_split" para manter o mesmo shuffle para todas as chamadas
RANDOM_STATE = 4
