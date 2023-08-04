import spacy
import pickle
import string
from sklearn.svm import SVC
import numpy as np
import os

# vocab = pickle.load(open('cache/vocab.p', 'rb'))
# word2idx = dict([(word,idx) for idx, word in enumerate(vocab)])
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

nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser','ner'])

def remove_punctuation(word):
    return ''.join([char for char in word if not char in string.punctuation+' '])

def tokenize(str_):
    tokens = [word.lemma_.lower() for word in nlp(str_) if not word.is_stop and word.lemma_.isalnum()]
    tokens = [word.replace('\n', '') for word in tokens if not word.isnumeric() and len(remove_punctuation(word.replace('\n', '')))!=0]
    return tokens

# nlp = spacy.load('en_core_web_lg', disable=['textcat','lemmatizer', 'parser', 'tagger','ner'])

# def remove_punctuation(word):
#     return ''.join([char for char in word if not char in string.punctuation+' '])

# def tokenize(str_):
#     tokens = [word.text.lower() for word in nlp(str_) if not word.is_stop]
#     tokens = [word.replace('\n', '') for word in tokens if not word.isnumeric() and len(remove_punctuation(word))!=0]
#     return tokens

from lxml import etree

import pickle
import os
import numpy as np
from numpy import load, save
import html
from bs4 import BeautifulSoup

class DataItem(object):
    vectors_path = '../04. Model of DP/precomputed/'
    UNK_LABEL = 'U'
    REL_LABEL = 'R'
    IREL_LABEL = 'I'    
    TYPE_GLOVE300 = 'G3'
    TYPE_GLOVE600 = 'G6'
    TYPE_BOW = 'B'
    GM1 = '/home/ec2-user/SageMaker/data/GM_all_1945_1956/'
    GM2 = '/home/ec2-user/SageMaker/data/GM_all_1957-1967/'
    
    GM1_SET = set([f[:-4] for f in os.listdir(GM1)])
    def __init__(self, *args):
        self._preloaded_vector={}
        if os.path.isfile(args[0]):
            file_ = args[0]
            self.id_ = file_.split('/')[-1][:-4]
            self.source = "GM1" if "GM_all_1945_1956" in file_ else "GM2"
            self.label=DataItem.UNK_LABEL
        else:
            id_ = args[0]
            if len(args)>1:
                source = args[1]
                assert source=="GM1" or source=="GM2"
            else:
                source = "GM1" if id_ in DataItem.GM1_SET else "GM2"
            assert id_.isnumeric()
            self.id_ = id_
            self.source = source
            self.label = DataItem.UNK_LABEL            
            assert os.path.isfile(self.filename())
    def preload_vector(self, type_=TYPE_BOW):
        if type_ not in self._preloaded_vector:
            self._preloaded_vector[type_] = self.vector(type_=type_)
#         if self._preloaded_vector is None:
#             self._preloaded_vector = self.vector(type_=type_)
    def get_htmldocview(self, highlighter=None):
        tree = etree.parse(self.filename())
        root = tree.getroot()
        if root.find('.//HiddenText') is not None:
            text = (root.find('.//HiddenText').text)

        elif root.find('.//Text') is not None:
            text = (root.find('.//Text').text)

        else:
            text = None
        title = root.find('.//Title').text
        date = root.find('.//NumericDate').text
        if not highlighter is None:
            text = html.unescape(text)
            text = BeautifulSoup(text, parser='html.parser',features="lxml").get_text()
            title = highlighter.highlight(title)
            text = highlighter.highlight(text)
#         for keyword in keywords:
#             text = re.sub(f'({keyword})', f'<mark>\\1</mark>', text, flags=re.IGNORECASE)
        # ADD DATE ########################
        url = f'https://proquest.com/docview/{self.id_}'
        url = f'<a href="{url}">{url}</a>'
        publisher = root.find('.//PublisherName').text
        return  '<html><hr style=\"border-color:black\">'\
                '<u>TITLE</u>: &emsp;&emsp;{}<br>'\
                '<u>DATE</u>: &emsp;&emsp;{}<br>'\
                '<u>PUBLISHER</u>: &emsp;{}<br>'\
                '<u>URL</u>:&emsp;&emsp;&emsp;{}<hr>'\
                '{}<hr style=\"border-color:black\"></html>'.format(
                                                                    str(title),
                                                                    date,
                                                                    publisher,
                                                                    url,
                                                                    str(text))
    
    def filename(self):
        filename = self.id_+'.xml'
        if self.source == "GM1":
            filename = DataItem.GM1+filename
        else:
            filename = DataItem.GM2+filename
        assert os.path.isfile(filename)
        return filename
    
    def has_vector(self):
        #_has_vector = os.path.isfile(self._vector_filename())
        _has_vector = os.path.isfile(self._vector_filename())
        
        return _has_vector
    
    def _vector_filename(self):
        return DataItem.vectors_path+self.id_+'_glove.p'
    
    def vector(self, type_=TYPE_BOW):
        if not type_ in self._preloaded_vector:
            assert type_==DataItem.TYPE_BOW or type_==DataItem.TYPE_GLOVE300 or type_ == DataItem.TYPE_GLOVE600
            assert os.path.isfile(self._vector_filename())

            data = pickle.load(open(self._vector_filename(), 'rb'))
            if type_==DataItem.TYPE_BOW:
                x = data[3].toarray()[0,:]
                if np.max(x)!=0:
                    x = x / np.max(x)
            elif type_==DataItem.TYPE_GLOVE300:
                x = data[0]
                assert x.shape==(300,)
            else:
                x = data[1]
                assert x.shape==(600,)
            return x
        else:
            return self._preloaded_vector[type_]
    
    def __str__(self):
        return f'DataItem(id={self.id_}, source={self.source}, label={self.label})'
    
    def get_X(item_list, type_=TYPE_BOW):
        assert all([item.has_vector() for item in item_list])
        X = np.zeros(shape=(len(item_list),item_list[0].vector(type_=type_).shape[0]),dtype='int32' if type_==DataItem.TYPE_BOW else 'float32')
        for idx,item in enumerate(item_list):
            X[idx,:] = item.vector(type_=type_)
        return X
    def get_y(item_list):
        assert all([item.label!=DataItem.UNK_LABEL for item in item_list])
        y = np.zeros(shape=(len(item_list),))
        for idx,item in enumerate(item_list):
            y[idx] = 1 if item.label==DataItem.REL_LABEL else 0
        return y
    
    def __eq__(self, other):
        return self.id_ == other.id_ and other.source == self.source

    def is_relevant(self):
        return self.label==DataItem.REL_LABEL
    def set_relevant(self):
        self.label = DataItem.REL_LABEL
    def set_irrelevant(self):
        self.label = DataItem.IREL_LABEL