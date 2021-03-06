import os
from lxml import etree
import pickle
import numpy as np
import html
from bs4 import BeautifulSoup
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


class DataItem(object):
    vectors_path = '/home/ec2-user/SageMaker/mariano/notebooks/04. Model of DP/precomputed/'
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
            title = highlighter.highlight(title)
            text = highlighter.highlight(text)

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
        filename = self.id_+'.xml'
        if self.source == "GM1":
            filename = DataItem.GM1+filename
        else:
            filename = DataItem.GM2+filename
        assert os.path.isfile(filename)
        return filename
    
    def has_vector(self):
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
            elif type_==DataItem.TYPE_GLOVE300:
                x = data[0]
                assert x.shape==(300,)
            else:
                x = data[1]
                assert x.shape==(600,)
               
            
            x = normalize([x])[0,:]
            return x
        else:
            return self._preloaded_vector[type_]
    
    def __str__(self):
        return f'DataItem(id={self.id_}, source={self.source}, label={self.label})'
    
    def get_X(item_list, type_=TYPE_BOW):
        assert all([item.has_vector() for item in item_list])
        
        X = np.zeros(shape=(len(item_list),item_list[0].vector(type_=type_).shape[0]),dtype='float32')
            
        for idx,item in enumerate(item_list):
            X[idx,:] = item.vector(type_=type_)
            
        if type_==DataItem.TYPE_BOW:
            X = csr_matrix(X)
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