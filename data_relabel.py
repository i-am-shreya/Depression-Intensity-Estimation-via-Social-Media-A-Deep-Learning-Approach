from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import word_tokenize
import numpy as np

# to load tokenizers/punkt/english.pickle
import nltk
nltk.download('punkt')

# to load sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt
import nltk
nltk.download('vader_lexicon')

# !pip install gensim
import gensim
from gensim.parsing.preprocessing import preprocess_documents
import gensim.downloader as api
model = api.load("glove-twitter-25")  # load glove vectors
b = model.most_similar('depression')
broad_terms = []
narrow_terms = []
for i in range(len(b)):
    if b[i][1] > 0.5:
        broad_terms.append(b[i][0])
    else:
        narrow_terms.append(b[i][0])
    

#load twitter data
import scipy.io as sio
train_data = sio.loadmat('train.mat')
# test_data = sio.loadmat('test.mat')

dep_scores = []
for i in range(len(train_data)):
    # Polarity
    sentences = []
    lines_list = tokenize.sent_tokenize(train_data[i])
    sentences.extend(lines_list)
    polarity = 0
    for sentence in sentences:
        sid = SentimentIntensityAnalyzer()
        print(sentence)
        ss = sid.polarity_scores(sentence)
        # for k in sorted(ss):
        #     print('{0}: {1}, '.format(k, ss[k]), end='')
        # print()
        polarity = polarity + ss['compound']
    
    #semantic score
    t = word_tokenize(train_data[i])
    h_b, h_n = 0, 0 # hit score
    for token in t:
        if token in broad_terms:
            h_b = h_b + 1
        elif token in narrow_terms:
            h_n = h_n + 1
        else:
            continue
    alpha = 0.8 #0.7 0.6
    semantic_score = (alpha*h_b) + ((1-alpha)*h_n)
    
    dep_score = polarity + semantic_score
    
    dep_scores.append(dep_score)
    
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

depression_scores = np.asarray(dep_scores)
depression_scores = depression_scores.reshape(-1,1)
#min-max normalization
depression_scores_scaled = min_max_scaler.fit_transform(depression_scores)


sio.savemat('train_label.mat', depression_scores_scaled)
# sio.savemat('test_label.mat', depression_scores_scaled)


