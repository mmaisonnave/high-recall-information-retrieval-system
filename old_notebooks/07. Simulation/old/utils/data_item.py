import os
from lxml import etree
from bs4 import BeautifulSoup
import pickle
import numpy as np
import html
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
from scipy import sparse
import spacy
import string

def get_vocab_path():
    return '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/MC_vocab.txt'
def get_idf_path():
    return '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/MC_idf.txt'

    
class DataItem(object):
    UNK_LABEL = 'U'
    REL_LABEL = 'R'
    IREL_LABEL = 'I'   
    
    TYPE_GLOVE300 = 'G3'
    TYPE_GLOVE600 = 'G6'
    TYPE_BOW = 'B'
    TYPE_HGGFC = 'HF'
    
    sources = ['/home/ec2-user/SageMaker/data/'+source for source in os.listdir('/home/ec2-user/SageMaker/data/')]
    
#     GM1 = '/home/ec2-user/SageMaker/data/GM_all_1945_1956/'
#     GM2 = '/home/ec2-user/SageMaker/data/GM_all_1957-1967/'
    
#     GM1_SET = set([f[:-4] for f in os.listdir(GM1)])



    vectors_paths = [
                     '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/precomputed/',
                     '/home/ec2-user/SageMaker/mariano/notebooks/04. Model of DP/precomputed/',
                    ]
    preloaded_extensions = [
                            '_representations.p',
                            '_glove.p',
                           ]
    def __init__(self, *args):
        self._preloaded_vector={}
        if os.path.isfile(args[0]):
            # When input is only a FILEPATH
            file_ = args[0]
            self.id_ = file_.split('/')[-1][:-4]
            self.source = '/'.join(file_.split('/')[:-1])
#             self.source = "GM1" if "GM_all_1945_1956" in file_ else "GM2"
            self.label=DataItem.UNK_LABEL    
            assert os.path.isfile(self.filename())
        else:
            # When input is ID
            self.id_ = args[0]
            if len(args)>1: 
                # when input is ID + source
                self.source = args[1]
#                 assert source=="GM1" or source=="GM2"
            else:
                # when input is only ID
#                 source = "GM1" if id_ in DataItem.GM1_SET else "GM2"
                for source in DataItem.sources:
                    if os.path.isfile(os.path.join(source,self.id_+'.xml')):
                        self.source=source
                        break
            assert self.id_.isnumeric(), f'id={id_} - type(id_)={type(id_)}'
#             self.id_ = id_
#             self.source = source
            self.label = DataItem.UNK_LABEL            
            assert os.path.isfile(self.filename())
            
    def preload_vector(self, type_=TYPE_BOW):
        if type_ not in self._preloaded_vector:
            self._preloaded_vector[type_] = self.vector(type_=type_)
            

    def _dict(self):
        return {'id': self.id_, 'source':self.source, 'label': self.label}
    
    def from_dict(dict_):
        item = DataItem(dict_['id'])        
        item.label = dict_['label']
        return item
        
    def get_vec_size(type_=TYPE_BOW):
        assert type_!= DataItem.TYPE_HGGFC, 'Hugging Face Models do not have a fixed vec size'
        if type_==DataItem.TYPE_BOW:
            return 10000
        elif type_==DataItem.TYPE_GLOVE300:
            return 300
        elif type_==DataItem.TYPE_GLOVE600:
            return 600
        

    
    def get_htmldocview(self, highlighter=None, confidence_score=None):
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
            title = highlighter.highlight(title, self)
            text = highlighter.highlight(text, self)

        if not confidence_score is None:
            confidence_score_str = f'{confidence_score:4.3f}'
        else:
            confidence_score_str = f' N/A '
        url = f'https://proquest.com/docview/{self.id_}'
        url = f'<a href="{url}">{url}</a>'
        publisher = root.find('.//PublisherName').text
        return  '<html><hr style=\"border-color:black\">'\
                '<u>TITLE</u>: &emsp;&emsp;{}<br>'\
                '<u>DATE</u>: &emsp;&emsp;{}<br>'\
                '<u>PUBLISHER</u>: &emsp;{}<br>'\
                '<u>URL</u>:&emsp;&emsp;&emsp;{}<br>'\
                '<u>CONFIDENCE SCORE</u>:&emsp;{}<hr>'\
                '{}<hr style=\"border-color:black\"></html>'.format(
                                                                    str(title),
                                                                    date,
                                                                    publisher,
                                                                    url,
                                                                    confidence_score_str,
                                                                    str(text))
    
    def filename(self):
        filename = os.path.join(self.source,self.id_+'.xml')
#         if self.source == "GM1":
#             filename = DataItem.GM1+filename
#         else:
#             filename = DataItem.GM2+filename
#         assert os.path.isfile(filename)
        return filename
    
    def has_vector(self):
        _has_vector = os.path.isfile(self._vector_filename())        
        return _has_vector
    
    def assign_label(self,label):
        assert label==DataItem.REL_LABEL or label==DataItem.IREL_LABEL
        if label==DataItem.REL_LABEL:
            self.set_relevant()
        else:
            self.set_irrelevant()
        
    def _vector_filename(self):
        for vector_path,extension in zip(DataItem.vectors_paths, DataItem.preloaded_extensions):
            if os.path.isfile(os.path.join(vector_path,self.id_+extension)):
                return os.path.join(vector_path,self.id_+extension)
#         return DataItem.vectors_path+self.id_+'_glove.p'
        assert False
    
    def vector(self, type_=TYPE_BOW):
        if not type_ in self._preloaded_vector:
            assert type_==DataItem.TYPE_BOW or type_==DataItem.TYPE_GLOVE300 or type_ == DataItem.TYPE_GLOVE600
            assert os.path.isfile(self._vector_filename())

            data = pickle.load(open(self._vector_filename(), 'rb'))
            if type(data)==list:
                if type_==DataItem.TYPE_BOW:
                    x = data[1].toarray()[0,:]
                elif type_==DataItem.TYPE_GLOVE300:
                    x = data[0]
                    assert x.shape==(300,)
                elif type_==DataItem.TYPE_HGGFC:
                    x = get_texts([self])
    #             else:
    #                 assert type_==DataItem.TYPE_GLOVE600
    #                 x = data[1]
    #                 assert x.shape==(600,)

                x = normalize([x])[0,:]
                return x
            elif type(data)==dict:
#                 return data['BoW'].toarray()[0,:]
                return data['BoW']
        else:
            return self._preloaded_vector[type_]
    
    def __str__(self):
        return f'DataItem(id={self.id_}, source={self.source}, label={self.label})'
    
    def get_X(item_list, type_=TYPE_BOW):
        assert all([item.has_vector() for item in item_list])
        if type_==DataItem.TYPE_HGGFC:
            return get_texts(item_list)
        else:
            X = sparse.vstack([item.vector() for item in item_list])
#             X = np.zeros(shape=(len(item_list),item_list[0].vector(type_=type_).shape[0]),dtype='float32')

#             for idx,item in enumerate(item_list):
#                 X[idx,:] = item.vector(type_=type_)

#             if type_==DataItem.TYPE_BOW:
#                 X = lil_matrix(X)
            return X
    
    def get_texts(item_list):
        texts = []
        for item in item_list:
            tree = etree.parse(item.filename())
            root = tree.getroot()
            if root.find('.//HiddenText') is not None:
                text = (root.find('.//HiddenText').text)

            elif root.find('.//Text') is not None:
                text = (root.find('.//Text').text)

            else:
                text = None
            title = root.find('.//Title').text
            concated_text = ''
            if not title is None:
                concated_text = f'{title}. '
            if not text is None:
                concated_text += f'{text}'
            texts.append(concated_text)                
        return texts
    def __hash__(self):
        return hash(f'{self.id_}-{self.source}')
    def get_y(item_list, type_=TYPE_BOW):
        assert all([item.label!=DataItem.UNK_LABEL for item in item_list])
        

        y = np.zeros(shape=(len(item_list),))
        for idx,item in enumerate(item_list):
            y[idx] = 1 if item.label==DataItem.REL_LABEL else 0
        
        if type_==DataItem.TYPE_HGGFC:
            new_y = np.zeros(shape=(len(y),2))
            for idx, arg in enumerate(y):
                new_y[idx,int(arg)]=1
            y = new_y
        return y
    
    def __eq__(self, other):
        return self.id_ == other.id_ and other.source == self.source

    def is_relevant(self):
        return self.label==DataItem.REL_LABEL
    def is_irrelevant(self):
        return self.label==DataItem.IREL_LABEL
    def is_unknown(self):
        return self.label==DataItem.UNK_LABEL
    def set_relevant(self):
        self.label = DataItem.REL_LABEL
    def set_irrelevant(self):
        self.label = DataItem.IREL_LABEL        
    def set_unknown(self):
        self.label = DataItem.UNK_LABEL
        
    def set_date(self, date):
        self.date=date
    def valid_date(self):
        return hasattr(self,'date')
    def get_date(self):
        assert self.valid_date()
        return self.date
    
    def set_confidence(self, confidence):
        self.date=confidence
    def valid_confidence(self):
        return hasattr(self,'confidence')
    def get_confidence(self):
        assert self.valid_confidence()
        return self.confidence
    
from utils.tokenizer import Tokenizer    
from scipy import sparse

class QueryDataItem(object):
#     vocab_path = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/vocab_with_dp.txt'
#     vocab_path = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/precomputed/vocab.txt'
#     vocab_path = 
#     idf_path = 
    idf = np.array([float(value) for value in open(get_idf_path(), 'r').read().splitlines()])
    vocab =  open(get_vocab_path(), 'r').read().splitlines()
    word2index = dict([(word,idx) for idx,word in enumerate(vocab)]) # dict([(linea.split(',')[1], int(linea.split(',')[0])) for linea in])
#     nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser','ner'])
    tokenizer = Tokenizer()
    
    def __init__(self, str_):
        self.text=str_
        self.label=DataItem.UNK_LABEL
        self.id_='synthetic'
    
    def _dict(self):
        return {'text': self.text, 'label':self.label}
    def from_dict(dict_):
        item = QueryDataItem(dict_['text'])        
        item.label = dict_['label']
        return item
    
#     def _remove_punctuation(word):
#         return ''.join([char for char in word if not char in string.punctuation+' '])

#     def _tokenize(str_):
#         tokens = [word.lemma_.lower() for word in QueryDataItem.nlp(str_) if not word.is_stop and word.lemma_.isalnum()]
#         tokens = [word.replace('\n', '') for word in tokens if not word.isnumeric()\
#                   and len(QueryDataItem._remove_punctuation(word.replace('\n', '')))!=0]
#         return tokens
    def vector(self, type_=DataItem.TYPE_BOW):
        assert type_==DataItem.TYPE_BOW, 'not implemented for other types'
        vector = np.zeros(shape=(QueryDataItem.get_vec_size(type_=type_),))
        
        token_list = QueryDataItem.tokenizer.tokenize(self.text)
        ngram_list = list(token_list)
        ngram_list += [' '.join(ngram) for ngram in Tokenizer.ngrams(ngram_list)]
        ngram_list = filter(lambda ngram: ngram in QueryDataItem.word2index , ngram_list)
        
        for token in ngram_list:
            if token in QueryDataItem.word2index:
                vector[QueryDataItem.word2index[token]]+=1
         

        vector = vector * QueryDataItem.idf
#         return normalize([vector,])[0,:]
    
        vector = normalize([vector,])[0,:]
        return lil_matrix(vector.reshape(1,-1))
    
    def has_vector(self):
        return True
    
    def get_vec_size(type_=DataItem.TYPE_BOW):
        assert type_== DataItem.TYPE_BOW, 'Not available for types differents than BoW'
        if type_==DataItem.TYPE_BOW:
            return 10000
        
    def is_relevant(self):
        return self.label==DataItem.REL_LABEL
    def is_irrelevant(self):
        return self.label==DataItem.IREL_LABEL
    def set_relevant(self):
        self.label = DataItem.REL_LABEL
    def set_irrelevant(self):
        self.label = DataItem.IREL_LABEL 
    def assign_label(self,label):
        assert label==DataItem.REL_LABEL or label==DataItem.IREL_LABEL
        if label==DataItem.REL_LABEL:
            self.set_relevant()
        else:
            self.set_irrelevant()
            
#     def _dict(self):
#         return {'text': self.text, 'label':self.label}
#     def from_dict(dict_):
#         item = QueryDataItem(dict_['text'])        
#         item.label = dict_['label']
#         return item
    
    def get_htmldocview(self):
        return  '<html><hr style=\"border-color:black\">'\
                '<hr>'\
                '{}<hr style=\"border-color:black\"></html>'.format(str(self.text))
    
    
class GenericSyntheticDocument:
    def __init__(self, str_, vocab_path=None, idf_path=None):
        self.text=str_
        self.label=DataItem.UNK_LABEL
        self.id_='synthetic'
        
        if not idf_path is None:
            self.idf = np.array([float(value) for value in open(idf_path, 'r').read().splitlines()])
        if not vocab_path is None:
            vocab =  open(vocab_path, 'r').read().splitlines()
            self.word2index = dict([(word,idx) for idx,word in enumerate(vocab)]) 
        self.tokenizer = Tokenizer()
        
    def vector(self):
        vector = np.zeros(shape=(self.get_vec_size(),))
        
        token_list = self.tokenizer.tokenize(self.text)
        ngram_list = list(token_list)
        ngram_list += [' '.join(ngram) for ngram in Tokenizer.ngrams(ngram_list)]
        ngram_list = filter(lambda ngram: ngram in self.word2index , ngram_list)
        
        for token in ngram_list:
            if token in self.word2index:
                vector[self.word2index[token]]+=1
         

        vector = vector * self.idf
        vector = normalize([vector,])[0,:]
        return lil_matrix(vector.reshape(1,-1))
    
    
    def assign_label(self,label):
        assert label==DataItem.REL_LABEL or label==DataItem.IREL_LABEL
        if label==DataItem.REL_LABEL:
            self.set_relevant()
        else:
            self.set_irrelevant()
            
            
    def is_relevant(self):
        return self.label==DataItem.REL_LABEL
    
    def _dict(self):
        return {'text': self.text, 'label':self.label, 'word2index':self.word2index, 'idf':list(self.idf)}
    def from_dict(dict_):
        item = QueryDataItem(dict_['text'])        
        item.label = dict_['label']
        item.word2index =  dict_['word2index']
        item.tokenizer = Tokenizer()
        item.idf = np.array(dict_['idf'])
        return item
    
    def set_relevant(self):
        self.label = DataItem.REL_LABEL
        
    def has_vector(self):
        return True
    def get_htmldocview(self):
        return  '<html><hr style=\"border-color:black\">'\
                '<hr>'\
                '{}<hr style=\"border-color:black\"></html>'.format(str(self.text))
    
    
    def get_vec_size(self):
        return len(self.word2index)
##########################################################################################
#                                         TESTER                                         #
##########################################################################################
def data_item_tester():
    item0 = DataItem('1287139125')
    assert item0.id_=='1287139125'
    assert item0.source=='GM1'
    assert item0.filename() == '/home/ec2-user/SageMaker/data/GM_all_1945_1956/1287139125.xml'
    assert item0.has_vector
    assert not item0.is_relevant()
    item0.set_irrelevant()
    assert not item0.is_relevant()
    item0.set_relevant()
    assert item0.is_relevant()    
    item0.set_irrelevant()
    assert not item0.is_relevant()
    
    vec = item0.vector(DataItem.TYPE_BOW)
    assert vec.shape==(10000,)
    vec = item0.vector(DataItem.TYPE_GLOVE300)
    assert vec.shape==(300,)
    vec = item0.vector(DataItem.TYPE_GLOVE600)
    assert vec.shape==(600,)
    print('OK!')
    
    item0.preload_vector(type_=DataItem.TYPE_BOW)
    assert len(item0._preloaded_vector)==1
    assert all(item0._preloaded_vector[DataItem.TYPE_BOW]==item0.vector(DataItem.TYPE_BOW))
    item0.preload_vector(type_=DataItem.TYPE_GLOVE300)
    assert len(item0._preloaded_vector)==2
    assert all(item0._preloaded_vector[DataItem.TYPE_GLOVE300]==item0.vector(DataItem.TYPE_GLOVE300))
    item0.preload_vector(type_=DataItem.TYPE_GLOVE600)
    assert len(item0._preloaded_vector)==3
    assert all(item0._preloaded_vector[DataItem.TYPE_GLOVE600]==item0.vector(DataItem.TYPE_GLOVE600))
    
    item0.get_htmldocview()
    
    item1 = DataItem('1269992885')
    assert item1.id_=='1269992885'
    assert item1.source=='GM2'
    
    
    item2 = DataItem('1269992885','GM2')
    
    assert item1==item2
    
    
    
    X = DataItem.get_X([item0, item1,item2],type_=DataItem.TYPE_BOW)
    assert X.shape==(3,10000)
    assert np.sum(X[0,:]-item0.vector(type_=DataItem.TYPE_BOW))<1e-5, np.sum(X[0,:]-item0.vector(type_=DataItem.TYPE_BOW))
    assert np.sum(X[1,:]-item1.vector(type_=DataItem.TYPE_BOW))<1e-5
    assert np.sum(X[2,:]-item2.vector(type_=DataItem.TYPE_BOW))<1e-5

    X = DataItem.get_X([item0, item1,item2],type_=DataItem.TYPE_GLOVE300)
    assert X.shape==(3,300)
    assert np.sum(X[0,:]-item0.vector(type_=DataItem.TYPE_GLOVE300))<1e-5
    assert np.sum(X[1,:]-item1.vector(type_=DataItem.TYPE_GLOVE300))<1e-5
    assert np.sum(X[2,:]-item2.vector(type_=DataItem.TYPE_GLOVE300))<1e-5
    
    
    X = DataItem.get_X([item0, item1,item2],type_=DataItem.TYPE_GLOVE600)
    assert X.shape==(3,600)
    assert np.sum(X[0,:]-item0.vector(type_=DataItem.TYPE_GLOVE600))<1e-5
    assert np.sum(X[1,:]-item1.vector(type_=DataItem.TYPE_GLOVE600))<1e-5
    assert np.sum(X[2,:]-item2.vector(type_=DataItem.TYPE_GLOVE600))<1e-5
    
    
    # get_y
    item0.set_irrelevant()
    item1.set_relevant()
    item2.set_irrelevant()
    
    y = DataItem.get_y([item0,item1,item2])
    assert all(y==[0, 1, 0])
    
    
    item0.set_relevant()
    item1.set_relevant()
    item2.set_irrelevant()
    y = DataItem.get_y([item0,item1,item2])
    assert all(y==[1, 1, 0])