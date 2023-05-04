import librosa  # https://librosa.org/
import librosa.display
import librosa.beat
import numpy as np
from scipy import spatial
import scipy.stats as st
import os


def estatistica(feature):
    media = np.mean(feature)
    desvio = np.std(feature)
    skewness = st.skew(feature)
    kurtosis = st.kurtosis(feature)
    mediana = np.median(feature)
    max = feature.max()
    min = feature.min()

    return np.array([media, desvio, skewness, kurtosis, mediana, max, min])


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


def spectral(statistic, line, y):

    mfcc = librosa.feature.mfcc(y=y, n_mfcc=13)
    # features[line][0] = mfcc
    for i in range(mfcc.shape[0]):
        statistic[line, i*7: i*7+7] = estatistica(mfcc[i, :])
        res = i*7

    spectral_centroid = librosa.feature.spectral_centroid(y=y)[0, :]
    # features[line][1] = spectral_centroid
    statistic[line, res: res+7] = estatistica(spectral_centroid)
    res += 7

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)[0, :]
    # features[line][2] = spectral_bandwidth
    statistic[line, res: res+7] = estatistica(spectral_bandwidth)
    res += 7

    spectral_contrast = librosa.feature.spectral_contrast(y=y)
    # features[line][3] = spectral_contrast
    for i in range(spectral_contrast.shape[0]):
        statistic[line, 105+i*7: 105 +
                  (i*7+7)] = estatistica(spectral_contrast[i, :])

    res = 154

    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0, :]
    # features[line][4] = spectral_flatness
    statistic[line, res: res+7] = estatistica(spectral_flatness)
    res += 7

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y)[0, :]
    # features[line][5] = spectral_rolloff
    statistic[line, res: res+7] = estatistica(spectral_rolloff)


def temporal(statistic, line, y, sr):

    res = 168

    f0 = librosa.yin(y=y, fmin=20, fmax=sr/2)
    f0[f0 == sr/2] = 0
    # features[line][6] = f0
    statistic[line, res: res+7] = estatistica(f0)
    res += 7

    rms = librosa.feature.rms(y=y)[0, :]
    # features[line][7] = rms
    statistic[line, res: res+7] = estatistica(rms)
    res += 7

    zero_cross = librosa.feature.zero_crossing_rate(y=y)[0, :]
    # features[line][8] = zero_cross
    statistic[line, res: res+7] = estatistica(zero_cross)


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
    # features = np.arange(9000, dtype=object).reshape((900, 10))
    statistic = np.zeros((900, 190), dtype=float)

    line = 0

    for name in nameList:
        name = str(name).replace("\"", "")
        fName = "Musics\\" + name + ".mp3"
        print("Musica nº: ", line+1, "Nome: ", name,)

        y, sr = librosa.load(fName, sr=22050, mono=True)

        # Spectral
        spectral(statistic, line, y)

        # Temporal
        temporal(statistic, line, y, sr)

        tempo = librosa.feature.tempo(y=y)
        statistic[line, 189] = tempo[0]

        line += 1

    statisticN = normalize(statistic)

    # Save
    np.savetxt("features_statistics.csv",
               statisticN, fmt="%lf", delimiter=",")


def metricas():

    # ============================ 3.1 ============================

    normalized = np.genfromtxt(
        "Features\\top100_normalized.csv", dtype=float, delimiter=",")

    statistics = np.genfromtxt("features_statistics.csv",
                               dtype=float, delimiter=",")

    euclidiana = np.zeros((900, 900), dtype=float)
    manhattan = np.zeros((900, 900), dtype=float)
    cosseno = np.zeros((900, 900), dtype=float)

    euclidiana_features = np.zeros((900, 900), dtype=float)
    manhattan_features = np.zeros((900, 900), dtype=float)
    cosseno_features = np.zeros((900, 900), dtype=float)

    # count = 0

    for i in range(900):

        # count += 1

        # print(count)

        for j in range(900):

            euclidiana[i][j] = np.linalg.norm(normalized[i] - normalized[j])
            manhattan[i][j] = np.linalg.norm(normalized[i] - normalized[j])
            cosseno[i][j] = spatial.distance.cosine(
                normalized[i], normalized[j])

            euclidiana_features[i][j] = np.linalg.norm(
                statistics[i] - statistics[j])
            manhattan_features[i][j] = np.linalg.norm(
                statistics[i] - statistics[j])
            cosseno_features[i][j] = spatial.distance.cosine(
                statistics[i], statistics[j])

    print("3.1 -> ✅")

    # ============================ 3.2 ============================

    np.savetxt("Features\\euclidiana.csv",
               euclidiana, fmt="%lf", delimiter=",")
    np.savetxt("Features\\euclidiana_features.csv",
               euclidiana_features, fmt="%lf", delimiter=",")

    np.savetxt("Features\\manhattan.csv", manhattan, fmt="%lf", delimiter=",")
    np.savetxt("Features\\manhattan_features.csv", manhattan_features,
               fmt="%lf", delimiter=",")

    np.savetxt("Features\\cosseno.csv", cosseno, fmt="%lf", delimiter=",")
    np.savetxt("Features\\cosseno_features.csv", cosseno_features,
               fmt="%lf", delimiter=",")

    print("3.2 -> ✅")

    return euclidiana, manhattan, cosseno, euclidiana_features, manhattan_features, cosseno_features


def read():

    # Read files
    euclidiana = np.genfromtxt(
        "Features\\euclidiana.csv", dtype=float, delimiter=",")
    euclidiana_features = np.genfromtxt(
        "Features\\euclidiana_features.csv", dtype=float, delimiter=",")

    manhattan = np.genfromtxt(
        "Features\\manhattan.csv", dtype=float, delimiter=",")
    manhattan_features = np.genfromtxt(
        "Features\\manhattan_features.csv", dtype=float, delimiter=",")

    cosseno = np.genfromtxt("Features\\cosseno.csv",
                            dtype=float, delimiter=",")
    cosseno_features = np.genfromtxt(
        "Features\\cosseno_features.csv", dtype=float, delimiter=",")

    top100 = np.genfromtxt(
        "Features\\top100_features.csv", dtype=str, delimiter=",")

    names = top100[1:, 0]

    for i in range(len(names)):

        names[i] = str(names[i]).replace("\"", "")

    return euclidiana, manhattan, cosseno, euclidiana_features, manhattan_features, cosseno_features, names


ficheiros = ["MT0000202045", "MT0000379144",
             "MT0000414517", "MT0000956340"]

nomes_lista = ["euclidiana", "manhattan", "cosseno",
               "euclidiana_features", "manhattan_features", "cosseno_features"]


def ranking_similaridade(names, nome_ficheiro, metric):

    start = np.where(
        names == nome_ficheiro)[0][0]

    ranking = np.argsort(metric[start, :])
    ranking = ranking[1:21]

    rank = 1

    # Go through list and nomes_lista at the same time

    for elem in ranking:

        print(rank, " -> ", names[elem] + ".mp3")

        rank += 1

        if elem == 20:
            break

    return ranking


def ranking_similaridadeV2(names, nome_ficheiro, metric):

    start = np.where(
        names == nome_ficheiro)[0][0]

    ranking = np.argsort(metric[start, :])
    ranking = ranking[1:21]

    rank = 1

    # Go through list and nomes_lista at the same time

    for elem in ranking:

        # print(rank, " -> ", names[elem] + ".mp3")

        rank += 1

        if elem == 20:
            break

    return ranking

# 3


def similaridade():

    euclidiana, manhattan, cosseno, euclidiana_features, manhattan_features, cosseno_features, names = read()

    # ============================ 3.3 ============================

    # Store the variables euclidiana, manhattan cosseno, euclidiana_features, manhattan_features and cosseno_features in a list
    list = [euclidiana, manhattan, cosseno, euclidiana_features,
            manhattan_features, cosseno_features]

    print("=================== RANKING SIMILARIDADE ===================\n")

    # For each file in Queries/
    for nome_ficheiro in ficheiros:

        print("================== FICHEIRO -> ",
              nome_ficheiro, "==================\n")

        for metric, nome in zip(list, nomes_lista):

            print("==========================",
                  nome, "==========================")

            ranking_similaridade(names, nome_ficheiro, metric)

        print("\n")


def ranking_metadata(names, nome_ficheiro, list):

    for metric in list:

        position = np.where(names == nome_ficheiro)[0][0]

        ranking = np.argsort(metric[position, :])

        rank = 0

        for elem in ranking:

            if (names[elem] != nome_ficheiro):

                # print(rank, "-> ", names[elem] + ".mp3")

                rank += 1

            if elem == 20:
                break

    return ranking


def metrica_precisa(ranking, rank_metadata):

    res = 0

    for elem in ranking:

        if elem in rank_metadata:

            res += 1

    return (res / 20) * 100


def count_Score():

    euclidiana, manhattan, cosseno, euclidiana_features, manhattan_features, cosseno_features, names = read()

    # ============================ 4.1 ============================

    # Read metadata from the second line
    metadados = np.genfromtxt("Dataset\\panda_dataset_taffc_metadata.csv", dtype=str,
                              delimiter=",")[1::, :]

    # metadados = np.genfromtxt(
    #     "Dataset\\panda_dataset_taffc_metadata.csv", dtype=str, delimiter=", ")

    score = np.zeros((900, 900))

    # count = 0

    for i in range(900):

        # count += 1
        # print(count)

        for j in range(900):

            result = 0

            if ((metadados[i][1].replace("\"", "") == metadados[j][1].replace("\"", "")) or (metadados[i][3].replace("\"", "") == metadados[j][3].replace("\"", ""))):
                result += 1

            for k in range(len(metadados[i][9].replace("\"", "").split("; "))):

                if metadados[i][9].replace("\"", "").split("; ")[k] in metadados[j][9].replace("\"", "").split("; "):
                    result += 1

            for k in range(len(metadados[i][11].replace("\"", "").split("; "))):

                if metadados[i][11].replace("\"", "").split("; ")[k] in metadados[j][11].replace("\"", "").split("; "):
                    result += 1

            score[i][j] = result

    # Add 1 to the diagonal
    np.fill_diagonal(score, 1)

    print("4.1 -> ✅")

    # ============================ 4.2 ============================
    np.savetxt("Features\\score.csv", score, fmt="%lf", delimiter=",")
    print("4.2 -> ✅")

    return score


def read_score():

    score = np.genfromtxt("Features\\score.csv", dtype=int, delimiter=",")

    return score


def metadata():

    euclidiana, manhattan, cosseno, euclidiana_features, manhattan_features, cosseno_features, names = read()

    # score = read_score()
    score = count_Score()

    # ============================ 4.3 ============================

    print("=========================== RANKING METADATA =================================\n")

    list = [euclidiana, manhattan, cosseno, euclidiana_features,
            manhattan_features, cosseno_features]

    for nome_ficheiro in ficheiros:

        print("================== FICHEIRO -> ",
              nome_ficheiro, " ==================\n")

        rank_metadata = ranking_metadata(names, nome_ficheiro, list)

        for metric, nome in zip(list, nomes_lista):

            print("==========================",
                  nome, "==========================")

            ranking = ranking_similaridadeV2(names, nome_ficheiro, metric)

            print(metrica_precisa(ranking, rank_metadata))

    print("\n")

    print("4.3 -> ✅")


def main():

    # 2.1
    # read_normalize()

    # 2.2 e 2.3
    # extract_features()

    # =============================================== 3 ===============================================

    # ============================ 3.1 e 3.2 ============================
    # euclidiana, manhattan, cosseno, euclidiana_features, manhattan_features, cosseno_features = metricas()

    # ============================ 3.3 ============================
    # euclidiana, manhattan, cosseno, euclidiana_features, manhattan_features, cosseno_features, names = similaridade()

    # =============================================== 4 ===============================================
    metadata()


if __name__ == "__main__":
    main()
