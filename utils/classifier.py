import re
from utils.data_item import DataItem
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
from sklearn.base import clone
class Classifier(object):
    
    def __init__(self, model, type_=DataItem.TYPE_BOW):
        self.model = model
        self.vector_type = type_
        self.trained=False
        self.rng = np.random.default_rng(2022)
        
    def _fit(self, item_list, partial=False):
        assert all([item.label!=DataItem.UNK_LABEL for item in item_list])

            
        X = DataItem.get_X(item_list, type_=self.vector_type)
        y = DataItem.get_y(item_list)
        if not partial:
            self.model = self.model.fit(X,y)
        else:
            self.model= self.model.partial_fit(X,y)
        self.trained=True
        
    def fit(self,item_list):
        self._fit(item_list,partial=False)
    
    def partial_fit(self, item_list):
        self._fit(item_list,partial=True)
        
    def predict(self,item_list):
        assert self.trained
        
        X = DataItem.get_X(item_list, type_=self.vector_type)
        
        return self.model.predict_proba(X)[:,1]
    def __str__(self):
        return f'<Clasifier vec_type={self.vector_type} trained={self.trained} model='+re.sub(' +','', str(self.model).replace('\n',''))+'>'
    
    def cross_validate_on(self, item_list,cv=5):
        X = DataItem.get_X(item_list, type_=self.vector_type)
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
    
from sklearn.neural_network import MLPClassifier


def test_classifier():
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
            
    for type_ in [DataItem.TYPE_BOW, DataItem.TYPE_GLOVE300, DataItem.TYPE_GLOVE600, ]:
        print(f'Working on {type_}')
        clf = Classifier(model=MLPClassifier(),type_=type_)
        assert not clf.trained
        clf.fit(labeled_data)
        yhat = clf.predict(labeled_data)

        clf.partial_fit(labeled_data)