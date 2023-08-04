import sys
sys.path.append('..')
import utils.high_recall_information_retrieval as hrir

system = hrir.HRSystem(from_scratch=False, 
                  session_name='default_simulation', 
                  finish_function=None, 
                  debug=False,
                  print_function=None)

import numpy as np
from sklearn.neural_network import MLPClassifier
from utils.classifier import Classifier
from utils.term_highlighter import TermHighlighter
from utils.data_item import DataItem

classifiers = [
#                 Classifier(MLPClassifier(early_stopping=True,
#                                          n_iter_no_change=20,
#                                          max_iter=1500,
#                                          hidden_layer_sizes=(20,), 
#                                          solver='adam', 
#                                          random_state=np.random.RandomState(42)), type_=DataItem.TYPE_BOW),
                Classifier(MLPClassifier(early_stopping=True,
                                         n_iter_no_change=20,
                                         max_iter=1500,
                                         hidden_layer_sizes=(100,), 
                                         solver='adam', 
                                         random_state=np.random.RandomState(42)), type_=DataItem.TYPE_GLOVE300),
              ]

term_highlighter = None

models = []
if not term_highlighter is None:
    models = [term_highlighter]
models += classifiers

def argrank_function(yhat):
    return list(reversed(np.argsort(yhat)))

import pandas as pd
if '__file__' in locals():
    print(__file__)
    output = '../simulation_results/' + __file__.split('/')[-1][:-3]+'.csv'
else:
    output = '../simulation_results/simulation_rsults_notebook.csv'

simulation_results = system.simulation(arg_rank_function=argrank_function,
                                       models=models,
                                       starting_no_of_examples=5,
                                       labeling_batch_size=10,
                                       random_state=np.random.default_rng(2022))

df = pd.DataFrame(simulation_results)
df.to_csv(output)