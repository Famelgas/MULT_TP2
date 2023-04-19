import librosa  # https://librosa.org/
import librosa.display
import librosa.beat
import numpy as np
import scipy.stats as st


def estatistica(feature):
    mean = np.mean(feature)
    desv = np.std(feature)
    skew = st.skew(feature)
    kurto = st.kurtosis(feature)
    median = np.median(feature)
    max_m = feature.max()
    min_m = feature.min()

    return np.array([mean, desv, skew, kurto, median, max_m, min_m])


def read_normalize():

    # Read file
    top100 = np.genfromtxt(
        "Features\\top100_features.csv", dtype=str, delimiter=",")
    top100 = top100[1::, 1:(len(top100[0])-1)].astype(float)

    # Normalize
    top100_n = np.zeros(top100.shape)

    for i in range(len(top100[0])):
        max_v = top100[:, i].max()
        min_v = top100[:, i].min()
        if (max_v == min_v):
            top100_n[:, i] = 0
        else:
            top100_n[:, i] = (top100[:, i] - min_v)/(max_v - min_v)

    # Save
    np.savetxt("Features\\top100_normalized.csv",
               top100_n, fmt="%lf", delimiter=",")


def extract_features():

    # Load file
    top100 = np.genfromtxt(
        "Features\\top100_features.csv", dtype=str, delimiter=",")
    nomes = top100[1:, 0]
    sr = 22050
    mono = True
    features = np.arange(9000, dtype=object).reshape((900, 10))
    stat = np.zeros((900, 190), dtype=float)

    line = 0

    for nome in nomes:
        nome = str(nome).replace("\"", "")
        fName = "Musics\\" + nome + ".mp3"
        print("Musica nº: ", line+1, "Nome: ", nome,)

        y, fs = librosa.load(fName, sr=sr, mono=mono)

        # Spectral
        mfcc = librosa.feature.mfcc(y=y, n_mfcc=13)
        features[line][0] = mfcc
        for i in range(mfcc.shape[0]):
            stat[line, i*7: i*7+7] = estatistica(mfcc[i, :])

        spectral_centroid = librosa.feature.spectral_centroid(y=y)[0, :]
        features[line][1] = spectral_centroid
        stat[line, 91: 91+7] = estatistica(spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)[0, :]
        features[line][2] = spectral_bandwidth
        stat[line, 98: 98+7] = estatistica(spectral_bandwidth)

        spectral_contrast = librosa.feature.spectral_contrast(y=y)
        features[line][3] = spectral_contrast
        for i in range(spectral_contrast.shape[0]):
            stat[line, 105+i*7: 105 +
                 (i*7+7)] = estatistica(spectral_contrast[i, :])

        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0, :]
        features[line][4] = spectral_flatness
        stat[line, 154: 154+7] = estatistica(spectral_flatness)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y)[0, :]
        features[line][5] = spectral_rolloff
        stat[line, 161: 161+7] = estatistica(spectral_rolloff)

        # Temporal
        f0 = librosa.yin(y=y, fmin=20, fmax=fs/2)
        f0[f0 == fs/2] = 0
        features[line][6] = f0
        stat[line, 168: 168+7] = estatistica(f0)

        rms = librosa.feature.rms(y=y)[0, :]
        features[line][7] = rms
        stat[line, 175: 175+7] = estatistica(rms)

        zero_cross = librosa.feature.zero_crossing_rate(y=y)[0, :]
        features[line][8] = zero_cross
        stat[line, 182: 182+7] = estatistica(zero_cross)

        # Other
        time = librosa.beat.tempo(y=y)
        stat[line, 189] = features[line][9] = time[0]

        line += 1

    # Normalize
    statisticN = np.zeros(stat.shape)
    for i in range(len(stat[0])):
        max_v = stat[:, i].max()
        min_v = stat[:, i].min()
        if (max_v == min_v):
            statisticN[:, i] = 0
        else:
            statisticN[:, i] = (stat[:, i] - min_v)/(max_v - min_v)

    # Save
    np.savetxt("features_statistics.csv",
               statisticN, fmt="%lf", delimiter=",")


def main():

    read_normalize()

    extract_features()


if __name__ == "__main__":
    main()
