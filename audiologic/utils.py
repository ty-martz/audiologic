import whisper

def score_acc(ytrue, ypred, print_confusion_matrix=False):
    '''
    Parameters:
        ytrue (): true class value
        ypred (): predicted class values
        print_confusion_matrix (bool): print scoring confusion matrix
    Return
        score
    '''

    assert len(ytrue) == len(ypred), f"ytrue and ypred are not the same length {len(ytrue)} != {len(ypred)}"
    # scoring
    if print_confusion_matrix:
        pass
    pass


def cv_test(data, labels, cv=5):
    '''
    Parameters
        data ():
        labels ():
        cv (int): number of splits in the cross validation testing
    Returns
        Summary of models
    '''

    assert len(data) == len(labels), f"Data and labels are not the same length {len(data)} != {len(labels)}"
    pass


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
