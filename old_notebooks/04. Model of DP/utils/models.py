import spacy
import pickle
import string
from sklearn.svm import SVC
import numpy as np
import os

vocab = pickle.load(open('cache/vocab.p', 'rb'))
word2idx = dict([(word,idx) for idx, word in enumerate(vocab)])
from threading import Lock
from utils.tdmstudio import TDMStudio

# import pandas as pd
# from lxml import etree
# from bs4 import BeautifulSoup

# def get_title_and_text(filename):
#     tree = etree.parse(filename)
#     root = tree.getroot()
#     if root.find('.//HiddenText') is not None:
#         text = (root.find('.//HiddenText').text)

#     elif root.find('.//Text') is not None:
#         text = (root.find('.//Text').text)

#     else:
#         text = None
                       
#     title = root.find('.//Title')
#     if title is not None:
#         title = title.text
#     if not text is None:
#         text = BeautifulSoup(text, parser='html.parser').get_text()

#     return title,text



def get_bow(title,text): 
#     title, text = TDMStudio.get_title_and_text(file_)
    x = np.zeros(shape=(len(vocab)+1,), dtype='float32')
    if not title is None and not text is None:
        tokens = tokenize(title+' '+text)
        for token in tokens:
            if token in word2idx:
                x[word2idx[token]]+=1
            else:
                x[-1]+=1
    return x
def get_glove300(title,text):
#     title, text = TDMStudio.get_title_and_text(file_)
    x = np.zeros(shape=(300,))
    if not title is None and not text is None:
        x = nlp(title+' '+text).vector
    return x

def get_glove600(title,text):
#     title, text = TDMStudio.get_title_and_text(file_)
    x = np.zeros(shape=(600,))
    if not title is None and not text is None:
        dtitle, dtext = nlp.pipe([title,text])
        x = np.zeros(shape=(600,), dtype='float32')
        x[:300] = dtitle.vector
        x[300:] = dtext.vector
    return x

def glove300_vectorize(titles, texts):
    X = np.zeros(shape=(len(texts),300))
    if not titles is None: 
        # Computing X values using Titles + Texts
        assert len(titles)==len(texts)
        assert all([not title is None for title in titles])
        assert all([not text is None for text in texts])
        idx=0
        
        for title,text in zip(titles,texts):
            X[idx,:]=nlp(f'{title}. {text}').vector
            idx+=1
#         for idx,doc in enumerate(nlp.pipe([f'{title}. {text}' for title,text in zip(titles,texts)])):
#             X[idx,:]=doc.vector 
#     else:  
#         # Computing X values using only Texts
# #         for idx,doc in enumerate(nlp.pipe(texts)):
#         for idx,text in enumerate(texts):
#             X[idx,:] = nlp(text).vector 
    return X
        
def _glove300_vectorize(texts):
    X = np.zeros(shape=(len(texts),300))
    for idx,text in enumerate(texts):
        X[idx,:] = nlp(text).vector 
    return X

def glove600_vectorize(titles, texts):
    assert len(titles)==len(texts)
    assert all([not title is None for title in titles])
    assert all([not text is None for text in texts])
    
    X = np.zeros(shape=(len(titles),600))
    idx=0
#     for dtitle,dtext in zip(nlp.pipe(titles),nlp.pipe(texts)):
    for title,text in zip(titles,texts):
        dtitle = nlp(title)
        dtext = nlp(text)
        X[idx,:300]=dtitle.vector    
        X[idx,300:]=dtext.vector   
        idx+=1
    return X

def best_paragraph(file_, vectorizers , models):
    assert os.path.isfile(file_)
    paragraphs = TDMStudio.get_paragraphs(file_)
    y = np.zeros(shape=(len(models),len(paragraphs)))
    idx=0
    for vectorize, model in zip(vectorizers, models):
        X = vectorize(paragraphs)
        y[idx] = model.predict_proba(X)[:,1]
        idx+=1
    yhat = np.average(y,axis=0)

    assert yhat.shape==(len(paragraphs),)
    return np.array(paragraphs)[np.argsort(yhat)][-1]

nlp = spacy.load('en_core_web_lg', disable=['textcat','lemmatizer', 'parser', 'tagger','ner'])

def remove_punctuation(word):
    return ''.join([char for char in word if not char in string.punctuation+' '])

def tokenize(str_):
    tokens = [word.text.lower() for word in nlp(str_) if not word.is_stop]
    tokens = [word.replace('\n', '') for word in tokens if not word.isnumeric() and len(remove_punctuation(word))!=0]
    return tokens
