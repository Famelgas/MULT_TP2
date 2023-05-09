import librosa  # https://librosa.org/
import librosa.display
import librosa.beat
import sounddevice as sd  # https://anaconda.org/conda-forge/python-sounddevice
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import spatial


def rank():

    musicas = [
        'MT0000202045', 'MT0000379144', 'MT0000414517', 'MT0000956340']

    topEuclidiana = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\top100_euclidiana.csv", dtype=float, delimiter=",")
    topManhattan = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\top100_manhattan.csv", dtype=float, delimiter=",")
    topCosseno = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\top100_cosseno.csv", dtype=float, delimiter=",")
    euclidianaF = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\features_euclidiana.csv", dtype=float, delimiter=",")
    manhattanF = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\features_manhattan.csv", dtype=float, delimiter=",")
    cossenoF = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\features_cosseno.csv", dtype=float, delimiter=",")
    top100_features = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\Features - Audio MER\\top100_features.csv", dtype=str, delimiter=",")

    name = top100_features[1:, 0]
    size = len(name)

    for i in range(size):
        name[i] = name[i].replace("\"", "")

    metadados = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\MER_audio_taffc_dataset\\panda_dataset_taffc_metadata.csv", dtype=str, delimiter=",")[1::, :]

    meta_mat = np.zeros((900, 900))

    for i in range(900):
        for j in range(i, 900):
            res = 0
            if (metadados[i][1].replace("\"", "") == metadados[j][1].replace("\"", "")):
                res += 1
            if (metadados[i][3].replace("\"", "") == metadados[j][3].replace("\"", "")):
                res += 1
            aux_1 = metadados[i][9].replace("\"", "").split("; ")
            aux_2 = metadados[j][9].replace("\"", "").split("; ")
            for k in range(len(aux_1)):
                if (aux_1[k] in aux_2):
                    res += 1

            aux_1 = metadados[i][11].replace("\"", "").split("; ")
            aux_2 = metadados[j][11].replace("\"", "").split("; ")

            for k in range(len(aux_1)):
                if (aux_1[k] in aux_2):
                    res += 1

            meta_mat[i][j] = res
            meta_mat[j][i] = res

    np.savetxt("C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\MER_audio_taffc_dataset\\similaridade.csv",
               meta_mat, fmt="%d", delimiter=",")

    for musica in musicas:

        print(musica)

        music_pos = np.where(name == musica)[0][0]

        eucl100 = np.argsort(topEuclidiana[music_pos, :])
        eucl100 = eucl100[1:21]
        man100 = np.argsort(topManhattan[music_pos, :])
        man100 = man100[1:21]
        cos100 = np.argsort(topCosseno[music_pos, :])
        cos100 = cos100[1:21]
        euclm = np.argsort(euclidianaF[music_pos, :])
        euclm = euclm[1:21]
        manm = np.argsort(manhattanF[music_pos, :])
        manm = manm[1:21]
        cosm = np.argsort(cossenoF[music_pos, :])
        cosm = cosm[1:21]

        print("\nRank do TOP100 Euclidiana\n")
        for pos in eucl100:
            print("'" + name[pos] + ".mp3'")

        print("\nRank do TOP100 Manhattan\n")
        for pos in man100:
            print("'" + name[pos] + ".mp3'")

        print("\nRank do TOP100 Cosseno\n")
        for pos in cos100:
            print("'" + name[pos] + ".mp3'")

        print("\nRank Euclidiano\n")
        for pos in euclm:
            print("'" + name[pos] + ".mp3'")

        print("\nRank Manhattan\n")
        for pos in manm:
            print("'" + name[pos] + ".mp3'")

        print("\nRank Cosseno\n")
        for pos in cosm:
            print("'" + name[pos] + ".mp3'")

        rank = meta_mat[music_pos, :].argsort()[::-1]

        print("\nRanking Metadados\n")

        rank_ = np.zeros(20)

        num = 0
        for pos in rank:
            if (name[pos] != musica):
                print("'" + name[pos] + ".mp3'")
                rank_[num] = pos
                num += 1
            if num == 20:
                break

        print(eucl100)
        print(rank_)
        print(cosm)

        eucl100n = np.intersect1d(eucl100, rank_)
        man100n = np.intersect1d(man100, rank_)
        cos100n = np.intersect1d(cos100, rank_)
        euclmn = np.intersect1d(euclm, rank_)
        manmn = np.intersect1d(manm, rank_)
        cosmn = np.intersect1d(cosm, rank_)

        print(eucl100n)
        print(cosmn)

        lista = []
        lista.append(len(eucl100n)/20.0)
        lista.append(len(man100n)/20.0)
        lista.append(len(cos100n)/20.0)
        lista.append(len(euclmn)/20.0)
        lista.append(len(manmn)/20.0)
        lista.append(len(cosmn)/20.0)

        print(lista)

        print("---------------------------------------------")


def distancias():
    # Implementação de métricas de similaridade.
    # Desenvolver o código Python/numpy para calcular as seguintes métricas de similaridade:

    top100_features = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\Features - Audio MER\\top100_normalized.csv", dtype=float, delimiter=",")
    statistic = np.genfromtxt(
        "C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\features_statistics.csv", dtype=float, delimiter=",")

    # print size of top100_features
    print(top100_features.shape)
    print(statistic.shape)

    # Nas diagonais os valores sao 0 porque estamos a calcular a distancia entre o mesmo ponto, por causa da matriz ser espelhada
    # As matrizes calculadas vem 900x900 porque para cada linha e comparada com as outras 899 linhas, dai o 900x900

    euclidiana = np.linalg.norm(
        top100_features[:, None] - top100_features, axis=2)

    manhattan = np.abs(top100_features[:, None] - top100_features).sum(axis=2)

    cosseno = 1 - np.dot(top100_features, top100_features.T) / (np.linalg.norm(
        top100_features, axis=1)[:, None] * np.linalg.norm(top100_features, axis=1))

    euclidianaF = np.linalg.norm(statistic[:, None] - statistic, axis=2)

    manhattanF = np.abs(statistic[:, None] - statistic).sum(axis=2)

    cossenoF = 1 - np.dot(statistic, statistic.T) / (np.linalg.norm(
        statistic, axis=1)[:, None] * np.linalg.norm(statistic, axis=1))

    # Criar e gravar em ficheiro 6 matrizes de similaridade (900x900)

    # Distância Euclidiana
    np.savetxt("C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\top100_euclidiana.csv",
               euclidiana, fmt="%lf", delimiter=",")
    np.savetxt("C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\features_euclidiana.csv",
               euclidianaF, fmt="%lf", delimiter=",")

    # Distância de Manhattan
    np.savetxt("C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\top100_manhattan.csv",
               manhattan, fmt="%lf", delimiter=",")
    np.savetxt("C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\features_manhattan.csv",
               manhattanF, fmt="%lf", delimiter=",")

    # Distância do Coseno
    np.savetxt("C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\top100_cosseno.csv",
               cosseno, fmt="%lf", delimiter=",")
    np.savetxt("C:\\Users\\andre\\Documents\\UC\\2ºS\\MULT\\TrabalhoMultimedia2\\res_ex3\\features_cosseno.csv",
               cossenoF, fmt="%lf", delimiter=",")


def estatistica(feature):
    media = np.mean(feature)
    devpad = np.std(feature)
    skewness = st.skew(feature)
    curtose = st.kurtosis(feature)
    mediana = np.median(feature)
    maxf = max(feature)
    minf = min(feature)

    return np.array([media, devpad, skewness, curtose, mediana, maxf, minf])


def main():

    # # Ler o ficheiro e criar um array numpy com as features disponibilizadas
    # top100 = np.genfromtxt(
    #     "Features - Audio MER\\top100_features.csv", dtype=str, delimiter=",")
    # top100 = top100[1::, 1:(len(top100[0])-1)].astype(float)

    # # Normalizar as features no intervalo [0, 1]
    # top100_n = np.zeros(top100.shape)

    # for i in range(len(top100[0])):
    #     max_v = top100[:, i].max()
    #     min_v = top100[:, i].min()
    #     if (max_v == min_v):
    #         top100_n[:, i] = 0
    #     else:
    #         top100_n[:, i] = (top100[:, i] - min_v)/(max_v - min_v)

    # # Criar e gravar em ficheiro um array numpy com as features extraídas (linhas = músicas; colunas = valores das features)

    # np.savetxt("Features - Audio MER\\top100_normalized.csv",
    #            top100_n, fmt="%lf", delimiter=",")

    # # Extrair features da framework librosa
    # # Para os 900 ficheiros da BD, extrair as seguintes features
    # # Load file

    # top100 = np.genfromtxt(
    #     "Features - Audio MER\\top100_features.csv", dtype=str, delimiter=",")
    # nomes = top100[1:, 0]
    # sr = 22050
    # mono = True
    # features = np.arange(9000, dtype=object).reshape((900, 10))
    # stat = np.zeros((900, 190), dtype=np.float64)

    # line = 0

    # for nome in nomes:
    #     nome = str(nome).replace("\"", "")
    #     fName = "FULLM\\" + nome + ".mp3"
    #     # para acompanhar o progresso
    #     print("Song number: ", line+1, "song: ", nome)

    #     y, fs = librosa.load(fName, sr=sr, mono=mono)

    #     # Features Espectrais

    #     mfcc = librosa.feature.mfcc(y=y, n_mfcc=13)
    #     features[line][0] = mfcc
    #     for i in range(mfcc.shape[0]):
    #         stat[line, i*7: i*7+7] = estatistica(mfcc[i, :])

    #     spectral_centroid = librosa.feature.spectral_centroid(y=y)[0, :]
    #     features[line][1] = spectral_centroid
    #     stat[line, 91: 91+7] = estatistica(spectral_centroid)

    #     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)[0, :]
    #     features[line][2] = spectral_bandwidth
    #     stat[line, 98: 98+7] = estatistica(spectral_bandwidth)

    #     spectral_contrast = librosa.feature.spectral_contrast(y=y)
    #     features[line][3] = spectral_contrast
    #     for i in range(spectral_contrast.shape[0]):
    #         stat[line, 105+i*7: 105 +
    #              (i*7+7)] = estatistica(spectral_contrast[i, :])

    #     spectral_flatness = librosa.feature.spectral_flatness(y=y)[0, :]
    #     features[line][4] = spectral_flatness
    #     stat[line, 154: 154+7] = estatistica(spectral_flatness)

    #     spectral_rolloff = librosa.feature.spectral_rolloff(y=y)[0, :]
    #     features[line][5] = spectral_rolloff
    #     stat[line, 161: 161+7] = estatistica(spectral_rolloff)

    #     # Features Temporais
    #     f0 = librosa.yin(y=y, fmin=20, fmax=fs/2)
    #     f0[f0 == fs/2] = 0
    #     features[line][6] = f0
    #     stat[line, 168: 168+7] = estatistica(f0)

    #     rms = librosa.feature.rms(y=y)[0, :]
    #     features[line][7] = rms
    #     stat[line, 175: 175+7] = estatistica(rms)

    #     zero_cross = librosa.feature.zero_crossing_rate(y=y)[0, :]
    #     features[line][8] = zero_cross
    #     stat[line, 182: 182+7] = estatistica(zero_cross)

    #     # Outras features
    #     time = librosa.beat.tempo(y=y)
    #     stat[line, 189] = features[line][9] = time[0]

    #     line += 1

    # # print(stat)

    # # Normalizar as features no intervalo [0, 1]
    # # np.savetxt("features_statistics.csv",
    # #            stat, fmt="%lf", delimiter=",")

    # statisticN = np.zeros(stat.shape)
    # for i in range(len(stat[0])):
    #     max_v = stat[:, i].max()
    #     min_v = stat[:, i].min()
    #     if (max_v == min_v):
    #         statisticN[:, i] = 0
    #     else:
    #         statisticN[:, i] = (stat[:, i] - min_v)/(max_v - min_v)

    # # Criar e gravar em ficheiro o array numpy com as features extraídas
    # np.savetxt("features_statistics.csv",
    #            statisticN, fmt="%lf", delimiter=",")

    # distancias()
    rank()


if __name__ == "__main__":
    main()
