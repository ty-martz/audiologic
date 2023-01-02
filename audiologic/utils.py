import whisper
import io
import librosa
from pydub import AudioSegment
from urllib.request import urlopen
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

def score_model(ytrue, ypred, metrics=['mae', 'rmse', 'r_squared']):
    '''
    Parameters:
        ytrue (): true class value
        ypred (): predicted class values
        metrics (str or list): metrics used to score model. Choose from MAE, RMSE, and R^2
    Return
        score
    '''

    assert len(ytrue) == len(ypred), f"ytrue and ypred are not the same length {len(ytrue)} != {len(ypred)}"
    # scoring
    scores = {}
    if ~isinstance(metrics, list):
        metrics = list(metrics)
    for met in metrics:
        if met == 'mae':
            mae = mean_absolute_error(ytrue, ypred)
            print(f"Mean Absolute Error = {mae}")
            scores['mae'] = mae
        elif met == 'rmse':
            rmse = mean_squared_error(ytrue, ypred)**0.5
            print(f"Mean Absolute Error = {rmse}")
            scores['rmse'] = rmse
        elif met == 'r_squared':
            r2 = r2_score(ytrue, ypred)
            print(f"R-Squared = {r2}")
            scores['r_squared'] = r2
        else:
            print(f'WARNING: Metric: {met} is not found. please select a subset of ')

    return scores


def cv_test(model, data, labels, cv=5):
    '''
    Parameters
        data ():
        labels ():
        cv (int): number of splits in the cross validation testing
    Returns
        Summary of models
    '''

    assert len(data) == len(labels), f"Data and labels are not the same length {len(data)} != {len(labels)}"
    crossval = cross_validate(model, data, labels, cv=cv, scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'], n_jobs=-1)
    print('----')
    print('CV Output:')
    print(f"Time to Fit: {crossval['fit_time']}")
    print(f"MAE: {-crossval['test_neg_root_mean_squared_error']}")
    print(f"RMSE: {-crossval['test_neg_mean_absolute_error']}")
    print(f"R-Squared: {crossval['test_r2']}")


def score_charts():
    pass


def transcribe_audio(audio_file, whisper_model='base', preloaded_model=None, english_only=False):
    '''
    Parameters:
        audio_file (str): path to file of audio data (.mp3, .wav, or other acceptable formats based on openai whisper)
        whisper_model (str): options include "tiny", "base", "small", "medium", "large" which get progressively slower/more accurate
        english_only (bool): to load the english only model or not
    '''
    if preloaded_model is None:
        if english_only:
            whisper_model = whisper_model + ".en"
        model = whisper.load_model(whisper_model)
    else:
        model = preloaded_model
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return result


def load_audio(file):
    if file[-4:] not in ['.mp3', '.wav']:
        wav = io.BytesIO()
        with urlopen(file) as r:
            r.seek = lambda *args: None  # allow pydub to call seek(0)
            AudioSegment.from_file(r).export(wav, "wav")
        wav.seek(0)
    elif file[-4:] == '.mp3':
        filepath = f'data/clips_45seconds/{file}'
        wav = io.BytesIO()
        AudioSegment.from_mp3(filepath).export(wav, "wav")
        wav.seek(0)
    elif file[-4:] == '.wav':
        wav = f'data/wav45/{file}'
    else:
        print('Check file type')

    y, sr = librosa.load(wav)
    return y, sr