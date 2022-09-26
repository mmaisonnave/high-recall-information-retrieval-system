from sklearn.linear_model import LogisticRegression
import numpy as np
import spacy
from utils.data_item import DataItem
from sklearn.preprocessing import normalize
from scipy import sparse
import pickle
import joblib
import json
from utils.data_item import QueryDataItem
import os

from tqdm import tqdm
from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
from sklearn.base import clone

from utils.tokenizer import Tokenizer

class TermHighlighter(object):
    def __init__(self, terms_per_article=10, keep_top=10000,):
        self.keep_top = keep_top
        self.terms_per_article=terms_per_article
        
        self.model = LogisticRegression(C=1, random_state=np.random.RandomState(42))
        self.trained = False
        self.rng = np.random.default_rng(2022)
        self.nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser','ner'])

        self.vocab=np.array(open(QueryDataItem.vocab_path, 'r').read().splitlines())
        self.tokenizer = Tokenizer()
    
    def to_disk(self, filename):
        configuration = {
            'keep_top':self.keep_top,
            'terms_per_article':self.terms_per_article,
            'trained':self.trained,
            }
        with open(f'{filename}.json','w') as outputfile:
            outputfile.write(json.dumps(configuration, indent=4))             
        joblib.dump(self.model, f'{filename}.joblib')
        
    def from_disk(filename):
        with open(f'{filename}.json', 'r') as f:
            configuration = json.load(f)
            
        term_highlighter = TermHighlighter()
        term_highlighter.trained = configuration['trained']
        term_highlighter.keep_top = configuration['keep_top']
        term_highlighter.terms_per_article = configuration['terms_per_article']
        
        term_highlighter.model = joblib.load(f'{filename}.joblib')
        return term_highlighter
    
    def fit(self, item_list):
        assert all([item.label!=DataItem.UNK_LABEL for item in item_list])
        
        X = DataItem.get_X(item_list, type_=DataItem.TYPE_BOW)
        y = DataItem.get_y(item_list)
#         self.mean_value_feature = np.average(X,axis=0)
        self.compute_mean_value_feature(item_list=None, X=X)
        self.model.fit(X,y)
        
        term_score = [(term,coef) for term,coef in zip(self.vocab, np.abs(self.model.coef_[0,:]))]
        term_score = sorted(term_score , key=lambda x:x[1],reverse=True)
        term_score = term_score[:self.keep_top]
        self.term2coef = dict(term_score)
        self.trained=True
    def compute_mean_value_feature(self, item_list=None, X=None):
        if X is None:
            assert not item_list is None
            X = DataItem.get_X(item_list, type_=DataItem.TYPE_BOW)
        self.mean_value_feature =np.average(X.toarray(),axis=0)
        
    def __str__(self):
        return f'<TermHighlighter model={self.model} trained={self.trained} vocab=<{self.vocab[0]}, ..., {self.vocab[1]}>>'
    
    
    def _optimized_logreg_prediction(self, item_list, batch_size=80530,progress_bar=False):
        
        linear_model = self.model
        vecnames = [item._vector_filename() for item in item_list]

        yhat = np.zeros(shape=(len(vecnames)))
        yhats = []
        iterable = range(0,len(vecnames), batch_size)
        if progress_bar:
            iterable=tqdm(iterable)
        for ini in iterable:
            fin = min(ini+batch_size, len(vecnames))
            X = normalize(sparse.vstack(map(lambda filename: pickle.load(open(filename, 'rb'))['BoW'], vecnames[ini:fin])),axis=1)
            X[:,:-2]=0
            yhats.append(linear_model.predict_proba(X)[:,1])
        return np.hstack(yhats)
    
    def predict(self,item_list, progress_bar=False):
        return self._optimized_logreg_prediction(item_list,progress_bar=progress_bar)
    
    def predict_old(self,item_list):
        assert self.trained
        max_mem_size_MB = 3072 # 3GB
#         vec = DataItem.get_X(item_list[:1], type_=DataItem.TYPE_BOW)
        vec_size = DataItem.get_vec_size(type_=DataItem.TYPE_BOW)
        instance_size_MB = (((vec_size*4)/1024)/1024)
        if instance_size_MB*len(item_list) <= max_mem_size_MB:
            X = DataItem.get_X(item_list, type_=DataItem.TYPE_BOW)
            return self.model.predict_proba(X)[:,1]
        else:
            yhat = np.zeros(shape=(len(item_list)))
            batch_size = int(max_mem_size_MB / instance_size_MB)
            for ini in range(0, len(item_list), batch_size):
                fin = ini+batch_size 
                X = DataItem.get_X(item_list[ini:fin], type_=DataItem.TYPE_BOW)
                yhat[ini:fin] = self.model.predict_proba(X)[:,1]
            return yhat
#         return self.model.predict_proba(X)[:,1]
    
    def highlight_coefficients(self,data_item):
        vec = data_item.vector(type_=DataItem.TYPE_BOW)
        coefs = self.model.coef_[0,:]
        mean_values = self.mean_value_feature
        result = coefs*(vec - mean_values)
        assert result.shape==(10000,)
        return result
    
    def highlight(self, text, data_item):
        term_score = [(term,coef) for term,coef in zip(self.vocab, self.highlight_coefficients(data_item) )]
#         term_score = sorted(term_score , key=lambda x:x[1],reverse=True)

#         term_score = term_score[:self.keep_top]
        term2coef = dict(term_score)
        
        
#         doc = self.nlp(text)
#         tokens = np.array([token for token in doc if token.lemma_.lower() in term2coef])

        tokens, indices = self.tokenizer.ngrams_and_indexes(text) # This includes uni-, bi-, and trigrams
        tokens = np.array([token for token in tokens if token in term2coef])
        indices = np.array([index for token,index in zip(tokens,indices) if token in term2coef])
        if len(tokens)>0:
            scores = np.array([term2coef[token] for token in tokens if token in term2coef])

            tokens = tokens[np.argsort(scores)][::-1]
            indices = indices[np.argsort(scores)][::-1]
        
        
        index_pairs = []
#         visited=set()
        for idx in range(len(tokens)):
            index_pairs.append((indices[idx][0],indices[idx][1], term2coef[tokens[idx]]))
#             index_pairs.append((tokens[idx].idx,
#                                 tokens[idx].idx+len(tokens[idx].text),
#                                 term2coef[tokens[idx].lemma_.lower()],
#                                ))
#             coefs.append()
#         idx=0
#         while idx<len(tokens):
#             visited.add(tokens[idx].lemma_.lower())
#             idx+=1
            
        return TermHighlighter._highlight(text, index_pairs)
#     def _overlap(pair1, pair2):
#         if pair1[0]==pair2[0]:
#             return True
#         elif pair1[0]<pair2[0]:
#             return not pair1[1]<=pair2[0]
#         else:
#             return not pair2[1]<=pair1[0]
        
#     def _longer(pair_list):
#         max_len = pair_list[0][1]-pair_list[0][0]
#         position=0
#         for idx,pair in enumerate(pair_list[1:]):
#             len_aux = pair[1] - pair[0]
#             if len_aux>max_len:
#                 position=idx+1
#                 max_len=len_aux
#         return pair_list[position]

    def _no_overlaping_in_list(index_pairs):
        flat_list = [elem for ini,fin,_ in sorted(index_pairs, key=lambda x:x[0]) for elem in [ini,fin]]
        return all(flat_list[i] <= flat_list[i+1] for i in range(len(flat_list) - 1))
    def _remove_overlaping(index_pairs,text):
        new_list=[]
        for elem in sorted(index_pairs,key=lambda x:x[2]):
            if TermHighlighter._no_overlaping_in_list(new_list+[elem]):
                new_list.append(elem)
        return sorted(list(set(new_list)), key=lambda x:x[0], reverse=True)
#         return new_list
#             assert all(flat_list[i] <= flat_list[i+1] for i in range(len(flat_list) - 1))
            
#         new_list=[]
#         if len(index_pairs)>0:
#             position =  min([ini for ini,fin,_ in index_pairs])
#             end = max([fin for ini,fin,_ in index_pairs])
# #             print(f'going from: {position} to {end}')
#             while position<end:
#                 contained_pairs = [(ini,fin,score) for ini,fin,score in index_pairs if ini<=position and position<fin]
# #                 print(f'Pairs found: {contained_pairs}')
#                 if len(contained_pairs)>0:
#                     best = sorted(contained_pairs, key=lambda x:x[2], reverse=True)[0] # Item with highest score
#                     new_list.append(best)
# #                     increment = max([fin for ini,fin,_ in contained_pairs])
#                     position=best[1]+1
# #                     print(f'New position {position} ({position-best[1]-1}+{best[1]}+1)')
#                 else:
#                     position+=1
# #                 if len(contained_pairs)>1:
# #                     valid_pairs = ';'.join([text[ini:fin] for ini,fin, _ in contained_pairs])
# #                     best_pair=text[best[0]:best[1]]
# #                     print(f'Replacing {valid_pairs} for {best_pair}.-')
# #                     print(f'Replacing '+';'.join([str((ini,fin)) for ini,fin,_ in contained_pairs])+' for '+str((best[0], best[1]) ))
#                 for item in contained_pairs:
#                     index_pairs.remove(item)
# #                     print(f'New position {position} ({position}+1)')
# #             print('end while')
# #             if len(new_list)>0:
# #                 position =  min([ini for ini,fin,_ in new_list])
# #                 end = max([fin for ini,fin,_ in new_list])
# #                 for position in range(position,end):
# #                     contained_pairs = [(ini,fin,score) for ini,fin,score in index_pairs if ini<=position and position<fin]
# #                     assert len(contained_pairs)<=1
# #                 print('process ok')
                    
#         return sorted(list(set(new_list)), key=lambda x:x[0], reverse=True)
    
    
    
#         for pair1 in index_pairs:
#             overlapping_pairs = [p for p in index_pairs if TermHighlighter._overlap(p,pair1) ]
#             new_list.append(TermHighlighter._longer(overlapping_pairs))
            
            
#             if len(overlapping_pairs)==1:
#                 new_list.append(pair1)
#             else:
    
    def _highlight(text, index_pairs):
        if len(index_pairs)>0:
#             for ini,fin,score in index_pairs:
#                 print(f'{text[ini:fin]:30} : {score:5.4f}')
#             print('### ### ### ### ### ### ### ### ### ### ### ### ###')
            
            index_pairs = sorted(index_pairs, key=lambda x:x[0], reverse=True)
            
#             len_old=len(index_pairs)

            index_pairs = TermHighlighter._remove_overlaping(index_pairs,text)
#             print(f'Size of index_pairs before: {len_old} and after removing overlapping {len(index_pairs)}')
#             if index_pairs!=old:
#                 with open('index_pairs.txt','a') as w:             
#                     w.write(text+'\n###\n')          
#                     w.write('\n'.join([str(p) for p in old]))
#                     w.write('\n-------------------------------------\n')
#                     w.write('\n'.join([str(p) for p in index_pairs])+'\n\n')

            flat_list = [elem for ini,fin,_ in sorted(index_pairs, key=lambda x:x[0]) for elem in [ini,fin]]
#             assert all(flat_list[i] <= flat_list[i+1] for i in range(len(flat_list) - 1))
        
        
            max_ = np.max(np.abs([coef for _,_,coef in index_pairs ]))
            if max_==0:
                max_=1
#             signs = np.array(coefs)>0
#             coefs = np.abs(coefs)
#             coefs = coefs/max(coefs)
            pos=0

            for ini,fin,coef in index_pairs:
                
#                 coef = coefs[pos]
#                 sign = signs[pos]
    #             full_color = 255 - 255*coef
                len_ = fin-ini
                if coef<=0:
                    middle_color = 255 - 145*((-coef)/max_)
                    text = text[:ini] +f'<mark style="background-color:rgb({255},{middle_color},{middle_color})">'+text[ini:fin]+'</mark>' +text[ini+len_:]
                else:
                    middle_color = 255 - 145*((coef)/max_)
                    text = text[:ini] +f'<mark style="background-color:rgb({middle_color},{255},{middle_color})">'+text[ini:fin]+'</mark>' +text[ini+len_:]
                pos+=1
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

    keep_top=100
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