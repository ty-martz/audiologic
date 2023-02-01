########
# Functions supporting the audio module in audiologic
########

# IMPORTS #
import whisper
import io
import librosa
import torch
import pickle
import numpy as np
import pandas as pd
from pydub import AudioSegment
from urllib.request import urlopen
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
import os.path

# MODEL PATH VARIABLES #
FIT_SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pred_scaler.pkl')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'tfidf_model.pkl')


def score_model(ytrue, ypred, metrics=['mae', 'rmse', 'r_squared']):
    '''
    Parameters:
        ytrue (array-like): true class value
        ypred (array-like): predicted class values
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
    Runs cross validation testing of a given model
    Parameters
        model : model class with which to test
        data (array-like): predictor variables
        labels (array-like): target variables
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
    '''
    custom function used in model training
    '''
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
    '''
    Loads audio files in 3 potential forms: .mp3, .wav, or url
    Parameters:
        file : path (mp3 or wav) or url to an audio file
    Returns:
        audio and sampling rate
    '''
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


def get_mel_spectrogram(file):
    '''
    Given a file path or url, returns the mel spectrogram data
    '''
    y, sr = load_audio(file)

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)

    # Normalize the spectrogram
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=64)
    mel_spectrogram = (mel_spectrogram + 100) / 100

    # Convert to PyTorch tensor
    mel_spectrogram = torch.from_numpy(mel_spectrogram).float()

    # Add a singleton dimension at the end to make the tensor 4D
    mel_spectrogram = mel_spectrogram.unsqueeze(0)

    # Add a singleton dimension at the beginning to make the tensor 4D
    mel_spectrogram = mel_spectrogram.unsqueeze(0)

    return mel_spectrogram


def is_array(var):
    '''checks if given variable is an array'''
    if np.ndim(var) != 0:
        return True
    else:
        # treat as single data point
        return False


def get_means(arr):
    '''returns the mean of an array and the mean of differences'''
    return [arr.mean(), np.diff(arr).mean()]


def feature_pipeline(file):
    '''
    given an audio file path, returns numerous features of the audio file
    '''
    # get features from audio data
    features = {} # empty dict for storing features
    audio, sample_rate = load_audio(file)

    # beat_tempo, diff
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
    features['tempo'] = tempo
    features['beat_length'] = len(beat_frames)
    features['beat_diff'] = np.mean([c-a for a, c in zip(beat_frames[:-1], beat_frames[1:])])
    
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    features['centroid'], features['d_centroid'] = [[x] for x in get_means(cent)]
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.95)
    features['rolloff'], features['d_rolloff'] = [[x] for x in get_means(rolloff)]
    
    rolloff_middle = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.5)
    features['rolloff_mid'], features['d_rolloff_mid'] = [[x] for x in get_means(rolloff_middle)]
    
    # Spectral Contrast
    S = np.abs(librosa.stft(y=audio))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sample_rate)
    for i, cont in enumerate(contrast):
        features[f"contrast_{i}"], features[f"d_contrast_{i}"] = [[x] for x in get_means(cont)]
        
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
    for i, mfcc in enumerate(mfccs):
        features[f"mfcc_{i}"], features[f"d_mfcc_{i}"] = [[x] for x in get_means(mfcc)]
    
    # RMS
    rms = librosa.feature.rms(y=audio)[0]
    features['rms'], features['d_rms'] = [[x] for x in get_means(rms)]
    
    df = pd.DataFrame(features)
    with open(FIT_SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    X = scaler.transform(df)

    return X


def lyric_pipeline(data):
    '''
    given an audio file, returns the tokenized lyrics of the sample
    '''
    model = whisper.load_model('base')
    with open(TOKENIZER_PATH, 'rb') as tok:
        tokenizer = pickle.load(tok)

    if is_array(data):
        assert len(data) > 0, f"Length of array should be more than 0"
        df = []
        lyrics = [transcribe_audio(i, preloaded_model=model) for i in data]
        df = [tokenizer.transform([x.text])[:1062] for x in lyrics]
        return df
    else:
        txt = transcribe_audio(data, preloaded_model=model)
        new_x = tokenizer.transform([txt.text]).toarray()
        return new_x[0][:1062].reshape(1, -1)


