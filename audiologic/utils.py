import whisper
import io
import librosa
import numpy as np
from pydub import AudioSegment
from urllib.request import urlopen
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_selection import SelectFromModel


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
            print(f"Root Mean Squared Error = {rmse}")
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
    print(f"Time to Fit: {np.mean(crossval['fit_time'])}")
    print(f"MAE: {np.mean(-crossval['test_neg_root_mean_squared_error'])}")
    print(f"RMSE: {np.mean(-crossval['test_neg_mean_absolute_error'])}")
    print(f"R-Squared: {np.mean(crossval['test_r2'])}")


def feature_testing(model, xtrain, ytrain, xval, yval, thresholds=[0.01, 0.025]):
    for thresh in thresholds:
        predictor_cols = ['tempo', 'beat_length', 'beat_diff',
                    'centroid', 'd_centroid', 'rolloff', 'd_rolloff', 'rolloff_mid',
                    'd_rolloff_mid', 'contrast_0', 'd_contrast_0', 'contrast_1',
                    'd_contrast_1', 'contrast_2', 'd_contrast_2', 'contrast_3',
                    'd_contrast_3', 'contrast_4', 'd_contrast_4', 'contrast_5',
                    'd_contrast_5', 'contrast_6', 'd_contrast_6', 'mfcc_0', 'd_mfcc_0',
                    'mfcc_1', 'd_mfcc_1', 'mfcc_2', 'd_mfcc_2', 'mfcc_3', 'd_mfcc_3',
                    'mfcc_4', 'd_mfcc_4', 'mfcc_5', 'd_mfcc_5', 'mfcc_6', 'd_mfcc_6',
                    'mfcc_7', 'd_mfcc_7', 'mfcc_8', 'd_mfcc_8', 'mfcc_9', 'd_mfcc_9', 'rms', 'd_rms']
        selector = SelectFromModel(estimator=model(), threshold=thresh).fit(xtrain, np.ravel(ytrain))
        xtrain_trans = selector.transform(xtrain)
        xval_trans = selector.transform(xval)
        mask = selector.get_support()
        cols = np.array(predictor_cols)[mask]
        newmod = model().fit(xtrain_trans, np.ravel(ytrain))
        predictions = newmod.predict(xval_trans)
        print(f"--Thresh={thresh} --> Droppped columns = {list(set(predictor_cols) - set(cols))}")
        scores = score_model(yval, predictions, ['mae'])


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