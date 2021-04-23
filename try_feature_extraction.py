import librosa
import numpy

import matplotlib.pyplot as plt
import librosa.display

import sklearn

genre = "pop"
num = "00005"

audio_file = "./genres/" + genre + "/" + genre + "." + num + ".wav"
x, sr = librosa.load(audio_file, sr=44100)



wave_plot = plt.figure(figsize=(13,5))
librosa.display.waveplot(x, sr=sr)
wave_plot.savefig("waveplot_" + genre + num +".png")
plt.close()


stft_data = librosa.stft(x)
stft_data_db = librosa.amplitude_to_db(abs(stft_data))
spectrogram = plt.figure(figsize=(13,5))
librosa.display.specshow(stft_data_db, sr=sr, x_axis="time", y_axis='hz')
plt.colorbar()
spectrogram.savefig("spectrogram_" + genre + num +".png")
plt.close()



spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids_plot = plt.figure(figsize=(13,5))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
librosa.display.waveplot(x, sr=sr, alpha=0.3)
plt.plot(t, sklearn.preprocessing.minmax_scale(spectral_centroids, axis = 0) , color='r')
spectral_centroids_plot.savefig("spectral_centroids_" + genre + num +".png")
plt.close()




spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01 , sr=sr)[0]
spectral_rolloff_plot = plt.figure(figsize=(13,5))
librosa.display.waveplot(x, sr=sr, alpha=0.3)
plt.plot(t, sklearn.preprocessing.minmax_scale(spectral_rolloff, axis = 0) , color='r')
spectral_rolloff_plot.savefig("spectral_rolloff_" + genre + num +".png")
plt.close()


























