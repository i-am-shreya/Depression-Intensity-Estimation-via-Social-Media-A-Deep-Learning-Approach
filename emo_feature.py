from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import word_tokenize
import numpy as np

# to load tokenizers/punkt/english.pickle
import nltk
nltk.download('punkt')

# to load sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt
import nltk
nltk.download('vader_lexicon')


#load twitter data
import scipy.io as sio
train_data = sio.loadmat('train_data.mat')
# test_data = sio.loadmat('test.mat')


#
#!pip install afinn
from afinn import Afinn
afinn = Afinn(language='en')

#emotion feature
emo_feature = np.zeros([len(train_data),9])
for i in range(len(train_data)):
    # LIWC
    liwc = afinn.score(train_data[i])
    
    # sentence level and word level
    analyzer = SentimentIntensityAnalyzer()
    ss = analyzer.polarity_scores(train_data[i])
    emo_sen = np.zeros(4)
    emo_sen[0] = ss['neg']
    emo_sen[1] = ss['neu']
    emo_sen[2] = ss['pos']
    emo_sen[3] = ss['compound']
    
    words = word_tokenize(train_data[i]) # data was pre-processed
    emo_word = np.zeros(4)
    for word in words:
        ss_ = analyzer.polarity_scores(word)
        emo_word[0] = emo_word[0] + ss_['neg']
        emo_word[1] = emo_word[1] + ss_['neu']
        emo_word[2] = emo_word[2] + ss_['pos']
        emo_word[3] = emo_word[3] + ss_['compound']
    
    emo_feature[i] = np.hstack((liwc,emo_sen,emo_word))
    
#load VAD feature
vad_data = sio.loadmat('VAD_data.mat')
emo_feature = np.hstack((emo_feature,vad_data))

sio.savemat('emo_feature.mat', emo_feature)
    
    
    
    
    