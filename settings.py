# TODO: Revisar essas configurações no final

# diretório com todas as músicas do GTZAN
GTZAN_DIRECTORY = "genres"

# directory with all musics tracks to train
TEST_MUSICS_DIRECTORY = "test"

# taxa de amostragem padrão das músicas da base GTZAN
SAMPLE_RATE = 22050

# nome do arquivo que contém os dados (features) das músicas
TRAIN_DATASET_PATH = "extracted_data_with_10_segments_and_30_sec.json"

# número a ser usado no "train_test_split" para manter o mesmo shuffle para todas as chamadas
RANDOM_STATE = 4

# teste dataset file name
TEST_DATASET_PATH = "extracted_data_with_10_segments_and_30_sec_test.json"
