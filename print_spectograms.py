import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def print_spectograms(file_path):
    y, sr = librosa.load(file_path, duration=15, sr=22050)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    M_db = librosa.power_to_db(M, ref=np.max)

    # No primeiro subplot exibe o espectograma STFT
    img1 = librosa.display.specshow(
        S_db, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='STFT (escala log)')
    ax[0].set(xlabel='Tempo')

    # No segundo subplot exibe o espectograma da escala mel
    img2 = librosa.display.specshow(
        M_db, x_axis='time', y_axis='mel', ax=ax[1])
    ax[1].set(title='Melspectograma')
    ax[1].set(xlabel='Tempo')

    fig.colorbar(img1, ax=ax[0], format="%+2.f dB")
    fig.colorbar(img2, ax=ax[1], format="%+2.f dB")

    plt.show()


if __name__ == '__main__':
    # Precisa passar um caminho válido de uma música ou um arquivo de exemplo, tipo 
    # print_spectograms("Guns N' Roses - Sweet Child O' Mine.mp3")
    print_spectograms(librosa.ex("choice"))
