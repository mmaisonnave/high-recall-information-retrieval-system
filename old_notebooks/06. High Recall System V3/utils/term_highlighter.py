from sklearn.linear_model import LogisticRegression
import numpy as np
import spacy
from utils.data_item import DataItem

from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
from sklearn.base import clone

class TermHighlighter(object):
    def __init__(self, terms_per_article=10, keep_top=20, vocab_path='/home/ec2-user/SageMaker/mariano/notebooks/04. Model of DP/precomputed/vocab_with_dp.txt'):
        self.keep_top = keep_top
        self.terms_per_article=terms_per_article
        
        self.model = LogisticRegression(C=1, random_state=np.random.RandomState(42))
        self.trained = False
        self.rng = np.random.default_rng(2022)
        self.nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser','ner'])

        self.vocab=np.array(open(vocab_path, 'r').read().splitlines())
        
    def fit(self, item_list):
        assert all([item.label!=DataItem.UNK_LABEL for item in item_list])
        
        X = DataItem.get_X(item_list, type_=DataItem.TYPE_BOW)
        y = DataItem.get_y(item_list)
        
        self.model.fit(X,y)
        
        term_score = [(term,coef) for term,coef in zip(self.vocab, np.abs(self.model.coef_[0,:]))]
        term_score = sorted(term_score , key=lambda x:x[1],reverse=True)
        term_score = term_score[:self.keep_top]
        self.term2coef = dict(term_score)
        self.trained=True
        
    def __str__(self):
        return f'<TermHighlighter model={self.model} trained={self.trained} vocab=<{self.vocab[0]}, ..., {self.vocab[1]}>>'
    def predict(self,item_list):
        assert self.trained
        
        X = DataItem.get_X(item_list, type_=DataItem.TYPE_BOW)
        return self.model.predict_proba(X)[:,1]
    
    def highlight(self, text):
        
        doc = self.nlp(text)
        tokens = np.array([token for token in doc if token.lemma_.lower() in self.term2coef])
        scores = np.array([self.term2coef[token.lemma_.lower()] for token in tokens])
        
        tokens = tokens[np.argsort(scores)][::-1]
        index_pairs = []
        visited=set()
        idx=0
        while len(visited)<self.terms_per_article and  idx<len(tokens):
            index_pairs.append((tokens[idx].idx,tokens[idx].idx+len(tokens[idx].text),))
            visited.add(tokens[idx].lemma_.lower())
            idx+=1
            
        return TermHighlighter._highlight(text, index_pairs)
    
    def _highlight(text, index_pairs):
        index_pairs = sorted(index_pairs, key=lambda x:x[0], reverse=True)
        for ini,fin in index_pairs:
            len_ = fin-ini
            text = text[:ini] +'<mark style="background-color:rgb(255,110,110)">'+text[ini:fin]+'</mark>' +text[ini+len_:]
        return text
    
    def sorted_terms(self):
        assert self.trained        
        return list(reversed((self.vocab[np.argsort(np.abs(self.model.coef_))])[0,:]))
    
    def cross_validate_on(self, item_list,cv=5):
        X = DataItem.get_X(item_list, type_=DataItem.TYPE_BOW)
        y = DataItem.get_y(item_list)
        scores = cross_validate(clone(self.model),
                                  X,
                                  y,
                                  cv=StratifiedKFold(n_splits=cv,shuffle=True,random_state=np.random.RandomState(2022)),
                                  scoring=['accuracy','precision','recall','f1'],
                                  return_train_score=True,
                                 ) 
        serie = pd.DataFrame(scores).mean()
        return dict([(metric,serie[metric]) for metric in serie.index])
    
##########################################################################################
#                                         TESTER                                         #
##########################################################################################
def test_term_highlighter():
    labeled_data=[]
    for line in open('labeled_data.csv').read().splitlines()[1:]:
        id_,label = line.split(';')
        item = DataItem(id_)
        if label=='R':
            item.set_relevant()
        else:
            item.set_irrelevant()
            assert label=='I'
        if item.has_vector():
            labeled_data.append(item)

    keep_top=20
    terms_per_article=10
    highlighter = TermHighlighter(keep_top=keep_top,terms_per_article=terms_per_article)
    assert not highlighter.trained
    assert len(highlighter.vocab)==10000
    
    highlighter.fit(labeled_data)
    
    assert len(highlighter.term2coef)==20
    assert len(highlighter.sorted_terms())==len(highlighter.vocab)
    
    assert all([elem1==elem2 for elem1, elem2 in zip(highlighter.term2coef, highlighter.sorted_terms()[:keep_top])])
    
    highlighter.highlight(' '.join(highlighter.sorted_terms()[:keep_top])) #first 'terms_per_articles' should be highlighted
    print('OK!')