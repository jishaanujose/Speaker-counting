# Speaker-counting
Count the number of speakers in an audio file

# Problem Statement
Speaker-attributed automatic speech recognition (SA-ASR) from overlapped speech has been an active research area towards meeting transcription. It requires to count the number of speakers, transcribe utterances that are sometimes overlapped, and also diarize or identify the speaker of each utterance. Speaker diarization is difficult when the number of speakers is large. If the number of speakers in an audio signal can be found, then the process of diarization can be made more simpler. 

# Solution
With the advancement in machine laerning systems, the number of speakers in an audio can be determined using a trained model. The speaker count predicted ny the model can be used to assist speaker diarization system for better transcription of the audio.

# Data description
The work uses VoxConverse dataset for the model creation. VoxConverse is an audio-visual diarisation dataset consisting of multispeaker clips of human speech, extracted from YouTube videos.It consists of over 50 hours of multispeaker clips of human speech. 

# Proposed model
A classification based approach is used in the project to identify the number of speakers in the audio. The work is limited to a single dataset and to three class: one speaker, two speakers and more than two speakers. 
The audio waves are initially preprocessed to remove noise and to separate speech from non speech part. Noise removal is done using spectral gating technique and is applied to a voice activity detector. The preprocessed signal is then given as input to YAMNet sound classification model. YAMNet (Yet Another MobileNet) is a pretrained acoustic detection model. 
It takes audio waveform as input and makes independent predictions by employing the MobileNet_v1 depth-wise-separable convolution architecture. The model will give scores, embeddings, and spectrograms as output. The embeddings from the YAMNet model is given to the speaker counting model. The work uses random forest classifier as the final stage model.  

# Tools used
The model is programmed using python language and uses frameworks such as numpy, scikit learn, and Flask. The project is demployed in Python3.6 in GoogleColab.

# Web app front
![speaker_front](https://user-images.githubusercontent.com/91037105/140272373-6a02c457-6f76-4e90-9ca8-05bb1cd65d33.png)
