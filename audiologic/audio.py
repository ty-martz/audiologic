# class and functions
# class = AudioModel
    # __init__
    # load_model
    # fit
    # predict
    # tuning
    # score


class AudioClassifier:

    def __init__(self, model_type='prefit'):

        self.model_type = model_type # prefit default for loading, "audio" to train on audio, "lyrics" to train on lyrics
        self.model = None
        self.predictions = None


    def load_model(self, model_choice: str):
        '''Load in an existing trained and tuned model'''
        if self.model_type != 'prefit':
            self.model_type = 'prefit'

        if model_choice == 'audio':
            pass
        elif model_choice == 'lyrics':
            pass


    def change_model(self, model_type: str):
        '''quickly change model after initializing the class'''
        self.model_type = model_type

    
    def fit_model(self, x, y, params):
        '''Given training audio and labels, fit a new model'''
        pass


    def tune_model(self, params):
        '''tune model with cross validation and scoring using an accuracy score'''
        pass


    def predict_model(self, data):
        '''given audio data, make predictions'''



