#import os
#from os import listdir
#import sys
#print(listdir('.'))


import unittest
import sys
sys.path.append('../audiologic')
import audiologic.audio as aud


class TestModel(unittest.TestCase):

    def test_audio_model(self):
        """
        Test prediction quality
        """

        model = aud.AudioClassifier('audio')
        data = 'https://p.scdn.co/mp3-preview/580c69cceaed0fda86c0d268343dd1f11fa5e1f5?cid=02b0e243b26741fa8d571a7f61aa4518'
        model.make_prediction(data)

        data2 = 'https://p.scdn.co/mp3-preview/ac967b5ecae24535339c511bcda5e518861e7b0f?cid=02b0e243b26741fa8d571a7f61aa4518'
        model.make_prediction(data2)

        #try: # values will change if model is refit
        #    self.assertEqual(model.audio_predictions[0], 5.34)
        #    self.assertEqual(model.audio_predictions[1], 4.76)
        #except:
        self.assertTrue(model.audio_predictions[0] != model.audio_predictions[1], "Audio model predicts same value")
        

if __name__ == '__main__':
    unittest.main()