########
# Main Module of audiologic. This allows the models to be used to predict valence of songs
########

# IMPORTS #
import pickle
import numpy as np
import audiologic.utils as util
import os.path

# PATH VARIABLES #
AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rf_audio_model.pkl')
LYRIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rf_lyric_model.pkl')


class AudioClassifier:

    def __init__(self, model_choice='both'):

        self.model_choice = model_choice

        if model_choice == 'audio':
            with open(AUDIO_MODEL_PATH, 'rb') as aud_file:
                self.audio_model = pickle.load(aud_file)
        elif model_choice == 'lyrics':
            with open(LYRIC_MODEL_PATH, 'rb') as lyr_file:
                self.lyric_model = pickle.load(lyr_file)
        elif model_choice == 'both':
            with open(AUDIO_MODEL_PATH, 'rb') as aud_file:
                self.audio_model = pickle.load(aud_file)
            with open(LYRIC_MODEL_PATH, 'rb') as lyr_file:
                self.lyric_model = pickle.load(lyr_file)

        self.audio_predictions = []
        self.lyric_predictions = []
        self.new_audio_features = []
        self.new_lyrics = []


    def get_last_prediction(self):
        '''
        Used to query the most recent prediction made by the model.
        Returns valence value or tuple of valence values if model_choice == 'both'
        '''

        if self.model_choice == 'audio':
            return self.audio_predictions[-1]
        elif self.model_choice == 'lyrics':
            return self.lyric_predictions[-1]
        elif self.model_choice == 'both':
            return (self.audio_predictions[-1], self.lyric_predictions[-1])
        else:
            raise AssertionError(f"model_choice must be (both, audio, lyrics), you chose {self.model_choice}")


    def refit_model(self, x, y):
        '''
        Update existing model with new data if enough new data available
        '''

        print('FUNCTION INCOMPLETE: Needs to be implemented')
        assert len(self.new_audio_features) >= 10, f"More than 10 new samples are required to refit the model, only {len(self.new_audio_features)} have been predicted on"
        pass


    def tune_model(self, params):
        '''
        tune model with cross validation and scoring using an accuracy score
        '''

        print('FUNCTION INCOMPLETE: Needs to be implemented')
        pass


    def audio_prediction_intervals(self, conf, new_val):
        '''
        get prediction interval based on model type. mainly used in the make_prediction
        function to get the confidence interval
        Parameters:
            conf (float): value between 0 ad 1 to set the confidence of the interval calculation
            new_val (): new value observation 
        Returns:
            (lo, hi) tuple of the interval
        '''

        trees = self.audio_model.estimators_
        tree_preds = []
        for m in trees:
            pred = m.predict(new_val)
            tree_preds.append(pred)
        samp_mean = np.mean(tree_preds)
        stdev = np.std(tree_preds)
        lo = samp_mean - (conf * (stdev / np.sqrt(len(tree_preds))))
        hi = samp_mean + (conf * (stdev / np.sqrt(len(tree_preds))))
        
        return (round(lo, 2), round(hi, 2))


    def make_prediction(self, data, include_ci=True, confidence=0.95):
        '''
        given single audio sample, make prediction on valence
        Parameters:
            data: new audio sample (mp3, wav, etc) to make a prediction on
            include_ci (bool): whether or not to include the confidence interval on the audio prediction
            confidence (float): value between 0 and 1, used if include_ci=True
        Returns:
            None, but prediction values are saved as class attributes
            and predictions are printed
        '''

        # load data in pipeline, predict in model
        def audio_predict():
            predictors = util.feature_pipeline(data)
            self.audio_predictions.append(round(self.audio_model.predict(predictors)[0], 2))

        def lyric_predict():
            predictors = util.lyric_pipeline(data)
            self.lyric_predictions.append(round(self.lyric_model.predict(predictors)[0], 2))

        if self.model_choice == 'audio':
            audio_predict()

        elif self.model_choice == 'lyrics':
            lyric_predict()

        elif self.model_choice == 'both':
            audio_predict()
            lyric_predict()
        
        else:
            raise AssertionError(f"No model has been trained or chosen. Current model choice = {self.model_choice}")

        print('----')
        if include_ci:
            ci = self.audio_prediction_intervals(confidence, util.feature_pipeline(data))
            print(f'Audio Valence = {self.audio_predictions[-1]} <> {100*confidence}% confidence interval of {ci[0]} - {ci[1]}')
            print(f'Lyrical Valence = {self.lyric_predictions[-1]}')
        else:
            print(f'Audio Valence = {self.audio_predictions[-1]}')
            print(f'Lyrical Valence = {self.lyric_predictions[-1]}')
        print('----')


