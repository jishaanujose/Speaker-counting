import tensorflow_hub as hub
# Load the YAMNET model.
model = hub.load('https://tfhub.dev/google/yamnet/1')

import os
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
from sklearn.decomposition import PCA
pca = PCA(n_components=2,random_state=42)
PATH = '/content/drive/MyDrive'
 # Define data path
data_path = PATH + '/speech'
data_dir_list = os.listdir(data_path)
featset=[]
labl=[]
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the audio of dataset-'+'{}\\n'.format(dataset))
    for i in range(0,len(img_list)):
      wav_fpath=data_path+'/'+ dataset+'/'+img_list[i]
      wav = preprocess_wav(wav_fpath)
      recov=reduce_noise(wav)
      mp3_file = AudioSegment.from_file(wav_fpath)
      mp3_file.export('newSong.wav', format="wav")
      wav_data, sample_rate1 = sf.read('newSong.wav', dtype=np.int16)
      sample_rate, wav_data = ensure_sample_rate(sample_rate1, recov)
      duration = len(wav_data)/sample_rate
      waveform = wav_data / tf.int16.max
      scores, embeddings, spectrogram = model(waveform)
      out=np.array(embeddings)
      t=pca.fit_transform(np.transpose(out))
      t=np.reshape(t,(t.shape[0]*t.shape[1]))
      featset.append(t)
      labl.append(dataset)
features=np.array(featset)
labels=np.array(labl)     

lab=np.zeros([labels.shape[0]],np.uint8)
for i in range(0,labels.shape[0]):
  if labels[i]=='Spe1':
    lab[i]=0
  elif labels[i]=='Spe2':
    lab[i]=1
  else:
    lab[i]=2
