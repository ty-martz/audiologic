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


def transcribe_audio(whisper_model='base', english_only=False):
    '''
    Parameters:
        whisper_model (str): options include "tiny", "base", "small", "medium", "large" which get progressively slower/more accurate
        english_only (bool): to load the english only model or not
    '''
    if english_only:
        whisper_model = whisper_model + ".en"
    model = whisper.load_model(whisper_model)
    pass
