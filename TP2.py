
import librosa
import librosa.display
import librosa.beat
import sounddevice as sd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import spatial



def stats(feature):
    mean = np.mean(feature)
    desv = np.std(feature)
    skew = st.skew(feature)
    kurto = st.kurtosis(feature)
    median = np.median(feature)
    max_m = feature.max()
    min_m = feature.min() 
    
    return np.array([mean, desv, skew, kurto, median, max_m, min_m])




def normalizar(arr):
    arr_normalizado = np.zeros(arr.shape)
    
    for i in range(len(arr[0])):
        max_v = arr[:, i].max()
        min_v = arr[:, i].min()
        if (max_v == min_v):
            arr_normalizado[:, i] = 0
        else:
            arr_normalizado[:, i] = (arr[:, i] - min_v)/(max_v - min_v)
        
    return arr_normalizado


def ex2(print):
    # 2.1
    #2.1.1
    top100_features = np.genfromtxt("Features - Audio MER/top100_features.csv", dtype = np.str, delimiter=",")
    fNames = top100_features[1:, 0]
    top100_features = top100_features[1::, 1:(len(top100_features[0])-1)].astype(np.float)
    
    if print == True:
        print(top100_features)
    
    #2.1.2
    top100_features_norm = normalizar(top100_features)
        
    if print == True:
        print(top100_features_norm)
   
    
    #2.1.3 
    np.savetxt("Features - Results/top100_features_normalized.csv", top100_features_norm, fmt = "%lf", delimiter= ",");
    
    #2.2.1
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    features= np.arange(9000, dtype=object).reshape((900,10))
    stats = np.zeros((900,190), dtype=np.float64)
    
    line = 0
    
    for name in fNames:
        name = name.replace("\"", "")
        fName = "MER_audio_taffc_dataset/music/" + name + ".mp3"
        print("Song number: ", line+1, "song: ", name)
        
        y, fs = librosa.load(fName, sr=sr, mono = mono)
        
        #2.2.2
        # espectrais
        mfcc = librosa.feature.mfcc(y=y,n_mfcc=13)
        features[line][0] = mfcc
        for i in range(mfcc.shape[0]):
            stats[line, i*7 : i*7+7] = stats(mfcc[i, :])
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y)[0,:]
        features[line][1] = spectral_centroid
        stats[line, 91 : 91+7] = stats(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)[0,:]
        features[line][2] = spectral_bandwidth
        stats[line, 98 : 98+7] = stats(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y)
        features[line][3] = spectral_contrast
        for i in range(spectral_contrast.shape[0]):
            stats[line, 105+i*7 : 105+(i*7+7)] = stats(spectral_contrast[i, :])
        
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0,:]
        features[line][4] = spectral_flatness
        stats[line, 154 : 154+7] = stats(spectral_flatness)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y)[0,:]
        features[line][5] = spectral_rolloff
        stats[line, 161 : 161+7] = stats(spectral_rolloff)
        
        # temporais
        f0 = librosa.yin(y=y, fmin=20, fmax=fs/2)
        f0[f0==fs/2]=0
        features[line][6] = f0
        stats[line, 168 : 168+7] = stats(f0)
        
        rms = librosa.feature.rms(y=y)[0,:]
        features[line][7] = rms
        stats[line, 175 : 175+7] = stats(rms)
        
        zero_cross = librosa.feature.zero_crossing_rate(y=y)[0,:]
        features[line][8] = zero_cross
        stats[line, 182 : 182+7] = stats(zero_cross)
        
        # outras
        time = librosa.beat.tempo(y=y)
        stats[line, 189] = features[line][9] = time[0]
    
        line += 1

    if print == True:
        print(stats)
    
    stats_normalizadas = normalizar(stats)
    
    np.savetxt("Features - Results/features_stats.csv", stats_normalizadas, fmt = "%lf", delimiter= ",");
    
    if print == True:
        print(stats_normalizadas)
        

if __name__ == "__main__":
    ex2(True)