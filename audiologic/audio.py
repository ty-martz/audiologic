import pickle
import numpy as np
from utils import *

# class and functions
# class = AudioModel
    # __init__
    # load_model
    # fit
    # predict
    # tuning
    # score


class AudioClassifier:

    def __init__(self, model_type='prefit', model_choice='audio'):

        self.model_type = model_type # prefit default for loading, "audio" to train on audio, "lyrics" to train on lyrics
        if model_type == 'prefit':
            self.model_choice = model_choice
            if model_choice == 'audio':
                self.model = pickle.loads('models/rf_audio_model.pkl')
            elif model_choice == 'lyrics':
                self.model = pickle.loads('models/rf_lyric_model.pkl')
            elif model_choice == 'both':
                self.model = pickle.loads('models/rf_audio_model.pkl')
                self.model2 = pickle.loads('models/rf_lyric_model.pkl')
        else:
            self.model = None
        self.predictions = None
        self.new_data = []


    #def load_model(self, model_choice: str):
    #    '''Load in an existing trained and tuned model'''
    #    if self.model_type != 'prefit':
    #        self.model_type = 'prefit'
    #
    #    if model_choice == 'audio':
    #        self.model = pickle.loads('models/rf_audio_model.pkl')
    #    elif model_choice == 'lyrics':
    #        self.model = pickle.loads('models/rf_lyric_model.pkl')


    def change_model(self, choice: str):
        '''quickly change model after initializing the class'''
        self.model_choice = choice
        if choice == 'audio':
            self.model = pickle.loads('models/rf_audio_model.pkl')
        elif choice == 'lyrics':
            self.model = pickle.loads('models/rf_lyric_model.pkl')
        elif choice == 'both':
            self.model = pickle.loads('models/rf_audio_model.pkl')
            self.model2 = pickle.loads('models/rf_lyric_model.pkl')

    
    def fit_model(self, x, y, params):
        '''Given training audio and labels, fit a new model'''
        pass


    def refit_model(self, x, y):
        '''Update existing model with new data if enough new data available'''
        assert len(self.new_data) >= 10
        pass


    def tune_model(self, params):
        '''tune model with cross validation and scoring using an accuracy score'''
        pass


    def predict_model(self, data):
        '''given audio data, make predictions'''
        # load data in pipeline, predict in model
        if self.model_choice == 'audio':
            predictors = feature_pipeline(data)
            preds = self.model.predict(data)

        elif self.model_choice == 'lyrics':
            data = lyric_pipeline(data)
            preds = self.model.predict(data)

        elif self.model_choice == 'both':
            predictors = feature_pipeline(data)
            self.model.predict(predictors)
            predictors = lyric_pipeline(data)
            preds = self.model2.predict(predictors)
        
        else:
            raise AssertionError(f"No model has been trained or chosen. Current model choice = {self.model_choice}")
        
        return preds



