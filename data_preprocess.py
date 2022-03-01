import re
import string
import numpy as np
import gensim
import spacy
from gensim.utils import simple_preprocess


import nltk;
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['http', 'https', 'www'])

#punctuation removal
def remove_punctuation(text):
  for p in string.punctuation:
    text = text.replace(p,'')
  return text

#pre-processing
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

#stopwords removal
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

#text Lemmatization with POS tagging
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#remove texts with consecutive characters
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

#convert the cleaned tweets back into string
def convert_to_string(df):
  for row in range(len(df)):
    df.iloc[row].Tweets = ' '.join([str(element) for element in df.iloc[row].Tweets])
  return df

def cleanTweets(df):
	
	# Remove new line characters
	df['Tweets'] = [re.sub('\s+', ' ', sent) for sent in df['Tweets']]
	# Remove Punctuations
	df['Tweets'] = df.Tweets.apply(remove_punctuation)
	# Remove distracting single quotes
	df['Tweets'] = [re.sub("\'", "", sent) for sent in df['Tweets']]
	# Remove consecutive characters
	df['Tweets'] = np.vectorize(remove_pattern)(df['Tweets'], "@[\w]*")

	df['Tweets'] = list(sent_to_words(df['Tweets']))
	df['Tweets'] = remove_stopwords(df['Tweets'])

	# Initialize spacy 'en' model
	df['Tweets'] = lemmatization(df['Tweets'], allowed_postags=['NOUN','ADJ','VERB','ADV'])

	# remove the stopwords again after lemmatizing the text
	df['Tweets'] = remove_stopwords(df['Tweets'])

	df = convert_to_string(df)
	df = df.drop([0], axis=0)
	return df


