import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def print_spectograms(file_path):
    y, sr = librosa.load(file_path, duration=5, sr=22050)
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    M_db = librosa.power_to_db(M, ref=np.max)

    # No primeiro subplot exibe o espectograma STFT
    img1 = librosa.display.specshow(
        S_db, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='STFT (escala log)')
    ax[0].set(xlabel=None)

    # No segundo subplot exibe o espectograma da escala mel
    img2 = librosa.display.specshow(
        M_db, x_axis='time', y_axis='mel', ax=ax[1])
    ax[1].set(title='Melspectograma')
    ax[1].set(xlabel=None)

    # No terceiro subplot exibe a formula de onda
    img3 = librosa.display.waveplot(y, sr=sr, ax=ax[2])
    ax[2].set(title='Waveform')
    ax[2].set(xlabel=None)
    ax[2].set(ylabel='Hz')

    fig.colorbar(img1, ax=ax[0], format="%+2.f dB")
    fig.colorbar(img2, ax=ax[1], format="%+2.f dB")
    fig.colorbar(img3, ax=ax[2], format="%+2.f dB")

    plt.show()


if __name__ == '__main__':
    # Precisa passar um caminho válido de uma música ou um arquivo de exemplo, tipo 
    # print_spectograms("songs/Guns N' Roses - Welcome To The Jungle.webm")
    print_spectograms(librosa.ex("choice"))
