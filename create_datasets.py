from song_features_extractor import extract_mfcc_from_multiple_songs
from settings import DATASET_PATH

if __name__ == '__main__':
    # cria os datasets com as features extraídas das músicas
    extract_mfcc_from_multiple_songs(DATASET_PATH, num_segments=5, track_duration=10)
    extract_mfcc_from_multiple_songs(DATASET_PATH, num_segments=5, track_duration=30)
    extract_mfcc_from_multiple_songs(DATASET_PATH, num_segments=10, track_duration=10)
    extract_mfcc_from_multiple_songs(DATASET_PATH, num_segments=10, track_duration=30)
    extract_mfcc_from_multiple_songs(DATASET_PATH, num_segments=15, track_duration=10)
    extract_mfcc_from_multiple_songs(DATASET_PATH, num_segments=15, track_duration=30)
