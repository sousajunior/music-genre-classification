from song_features_extractor import extract_mfcc_from_multiple_songs
from settings import GTZAN_DIRECTORY, NUMBER_OF_MFCCS, NUMBER_OF_SEGMENTS, TRACK_DURATION

if __name__ == '__main__':
    # cria os datasets com as features extraídas das músicas
    extract_mfcc_from_multiple_songs(
        GTZAN_DIRECTORY,
        num_mfcc=NUMBER_OF_MFCCS,
        num_segments=NUMBER_OF_SEGMENTS,
        track_duration=TRACK_DURATION
    )
