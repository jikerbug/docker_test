import tensorflow.keras as keras
import numpy as np
import librosa
import subprocess
import os
from os import path





MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050 # = 1 sec

class _Keyword_Spotting_Service:




    #singleton을 기초로 만든다....!

    model = None
    _instance = None
    _mappings = ["down","go","left","no","off","on","right","stop","up","yes"]


    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # coefficients)  (# : number of를 의미)

        # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, 1) (cnn은 3d여야 한다...!)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        preditioncs = self.model.predict(MFCCs) #[  [0.1, 0.6, 0.1, ...]  ]
        predicted_index = np.argmax(preditioncs)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):



        # load audio file
        signal, sr = librosa.load("Myself.mp4")

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T


def Keyword_Sporring_Service():

    #ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":

    kss = Keyword_Sporring_Service()



    keyword1 = kss.predict("test/Myself")
    #keyword2 = kss.predict("test/left.wav")

    print(f'Predicted keywords : {keyword1}')















