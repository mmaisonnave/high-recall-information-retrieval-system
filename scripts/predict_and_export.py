import re
import os
import pandas as pd
import numpy as np 
import time

import sys
sys.path.append('..')
from utils.data_item import DataItem
from utils.term_highlighter import TermHighlighter

##########
# SET UP #
##########
session_home = '/home/ec2-user/SageMaker/serperi/system/sessions/scal/One/'

labeled_data_file = os.path.join(session_home, 'data/labeled_data2022-10-11_19-40.csv')
log_file = os.path.join(session_home, 'log/scal_system.log')

assert os.path.isfile(labeled_data_file) and os.path.isfile(log_file)

################
# LABELED DATA #
################
labeled_data = {}
for id_, label in pd.read_csv(labeled_data_file, sep=';').values:
    id_=str(id_)
    assert label=='R' or label=='I'
    assert re.search('[0-9]{10}', id_)
    labeled_data[id_]={'label':label, 'data_item':DataItem(id_)}
    if label=='R':
        labeled_data[id_]['data_item'].set_relevant()
    else:
        labeled_data[id_]['data_item'].set_irrelevant()
        
##################
# MODEL TRAINING #
##################
model = TermHighlighter()
model.fit([labeled_data[id_]['data_item'] for id_ in labeled_data])
print(f'Model fitted with {len(labeled_data)} data items.')

t = float(re.search('Threshold\ *=([\.0-9]*)', open(log_file, 'r').read()).groups()[0])
print(f'Threshold={t}')

##################
# UNLABELED DATA #
##################
data_sources = ['/home/ec2-user/SageMaker/data/GM_not_all_1960_1978',
                '/home/ec2-user/SageMaker/data/GM_not_all_1979_1997',
                '/home/ec2-user/SageMaker/data/GM_not_all_1998_2018',
            ]

unlabeled_data = [DataItem(file.split('.')[0]) for data_source in data_sources for file in os.listdir(data_source)]
print(f'Size of corpus=   {len(unlabeled_data):,}')
unlabeled_data = list(filter(lambda item: not item.id_ in labeled_data, unlabeled_data))
print(f'Size of unlabeled={len(unlabeled_data):,}')

##############
# PREDICTION #
##############
# unlabeled_data = unlabeled_data[:750]
yhat = model.predict(unlabeled_data)
relevance = yhat>=t
print(f'Found {np.sum(relevance)} relevant.')
        
relevant_data=[labeled_data[id_]['data_item'] for id_ in labeled_data if labeled_data[id_]['data_item'].is_relevant()]
confidence = [1]*len(relevant_data)
no_of_labeled_rel = len(relevant_data)
print(f'Found {no_of_labeled_rel} relevant elements in labeled.')

for item,relevant,score in zip(unlabeled_data,relevance,yhat):
    if relevant:
        relevant_data.append(item)
        confidence.append(score)

print(f'With model suggestions now we have {len(relevant_data)} elements to export.')
print('Preparing file for exporting ...')
filename = os.path.join(session_home, f'data/exported_data_'+time.strftime("%Y-%m-%d_%H-%M")+'.csv')
# filename='final_predictions_serperi.txt'
with open(filename, 'w') as writer:
    writer.write('URL,relevant_or_suggested,confidence\n')
    count=0
    for item,confidence_value in zip(relevant_data,confidence):
        if count<no_of_labeled_rel:
            writer.write(f'https://proquest.com/docview/{item.id_},rel,{confidence_value:4.3f}\n')  
        else:
            writer.write(f'https://proquest.com/docview/{item.id_},sugg,{confidence_value:4.3f}\n')  
        count+=1

