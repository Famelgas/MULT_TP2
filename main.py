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
    top100_normalized = np.zeros(top100.shape)

    for i in range(len(top100[0])):
        max_v = top100[:, i].max()
        min_v = top100[:, i].min()
        if (max_v == min_v):
            top100_normalized[:, i] = 0
        else:
            top100_normalized[:, i] = (top100[:, i] - min_v)/(max_v - min_v)

    # Save
    np.savetxt("Features\\top100_normalized.csv",
               top100_normalized, fmt="%lf", delimiter=",")


def spectral(features, statistic, line, y):

    mfcc = librosa.feature.mfcc(y=y, n_mfcc=13)
    features[line][0] = mfcc
    for i in range(mfcc.shape[0]):
        statistic[line, i*7: i*7+7] = estatistica(mfcc[i, :])

    spectral_centroid = librosa.feature.spectral_centroid(y=y)[0, :]
    features[line][1] = spectral_centroid
    statistic[line, 91: 91+7] = estatistica(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)[0, :]
    features[line][2] = spectral_bandwidth
    statistic[line, 98: 98+7] = estatistica(spectral_bandwidth)

    spectral_contrast = librosa.feature.spectral_contrast(y=y)
    features[line][3] = spectral_contrast
    for i in range(spectral_contrast.shape[0]):
        statistic[line, 105+i*7: 105 +
                  (i*7+7)] = estatistica(spectral_contrast[i, :])

    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0, :]
    features[line][4] = spectral_flatness
    statistic[line, 154: 154+7] = estatistica(spectral_flatness)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y)[0, :]
    features[line][5] = spectral_rolloff
    statistic[line, 161: 161+7] = estatistica(spectral_rolloff)


def temporal(features, statistic, line, y, sr):

    f0 = librosa.yin(y=y, fmin=20, fmax=sr/2)
    f0[f0 == sr/2] = 0
    features[line][6] = f0
    statistic[line, 168: 168+7] = estatistica(f0)

    rms = librosa.feature.rms(y=y)[0, :]
    features[line][7] = rms
    statistic[line, 175: 175+7] = estatistica(rms)

    zero_cross = librosa.feature.zero_crossing_rate(y=y)[0, :]
    features[line][8] = zero_cross
    statistic[line, 182: 182+7] = estatistica(zero_cross)


def normalize(statistic):

    # Normalize
    statisticN = np.zeros(statistic.shape)
    for i in range(len(statistic[0])):
        max_v = statistic[:, i].max()
        min_v = statistic[:, i].min()
        if (max_v == min_v):
            statisticN[:, i] = 0
        else:
            statisticN[:, i] = (statistic[:, i] - min_v)/(max_v - min_v)

    return statisticN


def extract_features():

    # Load file
    top100 = np.genfromtxt(
        "Features\\top100_features.csv", dtype=str, delimiter=",")
    nameList = top100[1:, 0]
    features = np.arange(9000, dtype=object).reshape((900, 10))
    statistic = np.zeros((900, 190), dtype=float)

    line = 0

    for name in nameList:
        name = str(name).replace("\"", "")
        fName = "Musics\\" + name + ".mp3"
        print("Musica nº: ", line+1, "Nome: ", name,)

        y, sr = librosa.load(fName, sr=22050, mono=True)

        # Spectral
        spectral(features, statistic, line, y)

        # Temporal
        temporal(features, statistic, line, y, sr)

        tempo = librosa.feature.tempo(y=y)
        statistic[line, 189] = features[line][9] = tempo[0]

        line += 1

    statisticN = normalize(statistic)

    # Save
    np.savetxt("features_statistics.csv",
               statisticN, fmt="%lf", delimiter=",")


def main():

    read_normalize()

    extract_features()


if __name__ == "__main__":
    main()
