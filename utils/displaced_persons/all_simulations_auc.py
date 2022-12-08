import os
import pandas as pd
import pandas
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('/home/ec2-user/SageMaker/mariano/repositories/tdmstudio-high-recall-information-retrieval-system/')

from utils.newsgroup20.scal import SCAL20NG
from utils.newsgroup20.dataset import Dataset20NG, DataItem20NG

from utils.io import info, warning

from sklearn.metrics import auc

def how_many_processed(df, N, n, model, representation, rank_function, category):
    mask = (df['N']==N) & (df['n']==n) & (df['representation']==representation) \
                                & (df['Model']==model) & (df['Ranking Function']==rank_function)
    return np.sum(mask)

def main():
    
    # ~INPUT ~
    results_file = '/home/ec2-user/SageMaker/mariano/datasets/displaced_persons/simulation_results/all_results_final.csv'
    
    # ~OUTPUT ~
    output_file = '/home/ec2-user/SageMaker/mariano/datasets/displaced_persons/simulation_results/auc_results.csv'
    
    target_recall=0.8
     # CHANGE <<<<<<<<<    
    
    dataset = 'newsgroup20'
    categories = os.listdir('/home/ec2-user/SageMaker/mariano/datasets/20news-18828/files/')
    dataset_size=18828
    Ns = [int(p*dataset_size) for p in [0.05, 0.10, 0.20, 0.25, 0.5, 0.75, 1.0]]
    ns = [1, 3, 5, 10, 20]
    models = ['logreg', 'svm']
    representations = ['bow', 'sbert', 'glove']
    sampling_functions = ['relevance', 
                          'uncertainty',
                          '1quarter_relevance_3quarters_uncertainty',
                          '3quarter_relevance_1quarters_uncertainty',
                          'relevance_with_avg_diversity',
                          'relevance_with_min_diversity',
                          'uncertainty_with_avg_diversity',
                          'uncertainty_with_min_diversity',
                          'half_relevance_half_uncertainty']
    
    assert os.path.isfile(results_file)
    df = pd.read_csv(results_file)
    info(f'Read previous results. Results found: {len(df):,}')
    
    total=0
    data = {'representation': [],
            'Model': [],
            'Ranking Function': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
           }
    rows=0
    no_of_efforts=len(set(df['Effort']))
    for representation in representations:
        for model in models:
            for rank_function in sampling_functions:
                mask = (df['Model']==model) & (df['representation']==representation) & (df['Ranking Function']==rank_function)
#                 assert df[mask].shape[0]==len(Ns)* len(ns)*len(categories)
                results = df[mask].groupby('Effort').mean()
#                 assert results.shape[0]==no_of_efforts
                if results.shape[0]==no_of_efforts:   
                    rows+=1                 
                    print(f'results.shape={results.shape}')
                    data['representation']+=[representation]
                    data['Model']+=[model]
                    data['Ranking Function']+=[rank_function]
                    results=results.sort_index()
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                        x=results.index
                        y=results[metric]
                        auc_score = auc(x,y) / auc(x,[1]*len(x))

                        data[metric].append(auc_score)
#                 else:
#                     print(f'results.shape={results.shape}')
#                     print(f'results.columns={results.columns}')
#                     print(f'results[Ns]='+str(results['N']))
#                     print(f'results[ns]='+str(results['n']))
#                     print(f'results.shape={results.shape}')
                    
    info(f'Rows computed: {rows}')            
    info(f'Saving AUC results to: {output_file}')
    pd.DataFrame(data).to_csv(output_file)
                
if __name__=='__main__':
    main()