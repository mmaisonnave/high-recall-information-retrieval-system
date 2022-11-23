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

import math
def effort(N,n):
    B=1
    effort=0
    while N>0:
        b = min(n,B)
        N-=B
        effort+=b
        B+=math.ceil(B/10)
    return effort
def how_many_processed(df, N, n, model, representation, rank_function, category):
    mask = (df['N']==N) & (df['n']==n) & (df['representation']==representation) \
                                & (df['Model']==model) & (df['Ranking Function']==rank_function) & (df['category']==category)
    return np.sum(mask)

if __name__=='__main__':
    target_recall=0.8
     # CHANGE <<<<<<<<<
    dataset = 'newsgroup20'
    categories = os.listdir('/home/ec2-user/SageMaker/mariano/datasets/20news-18828/files/')
    dataset_size=18828
    no_of_seeds = 1
    Ns = [int(p*dataset_size) for p in [0.05, 0.10, 0.20, 0.25, 0.5, 0.75, 1.0]]
    ns = [1, 3, 5, 10, 20]
    models = ['logreg', 'svm']
    representations = ['bow', 'sbert', 'glove']
    sampling_functions = ['relevance',]
    
    results_file = '/home/ec2-user/SageMaker/mariano/datasets/20news-18828/simulation_results/all_results_v3.csv'
    df=None
    if os.path.isfile(results_file):
        df = pd.read_csv(results_file)
        info(f'Read previous results. Results found: {len(df):,}')
    
    total=0
    info(f'Total of repetitions requested: {len(categories)*len(representations)*len(models)*len(Ns)*len(ns)*no_of_seeds:,}')
    for category in categories:
        info(f'Working with category={category}')
#         oracle = Dataset20NG.get_20newsgroup_oracle(category=category)
        for representation in representations:
            info(f'Working with representation={representation}')
#             representations = Dataset20NG.get_20newsgroup_representations(type_=representation)
            for model in models:
                info(f'MODEL={model}')
                for rank_function in sampling_functions:
                    for N in tqdm(Ns):
                        for n in ns:
                            to_do=no_of_seeds
                            if not df is None:                            
                                processed = how_many_processed(df, N, n, model, representation, rank_function, category)
                                if processed>0:
                                    pass
                                if to_do<processed:
                                    warning('Found more repetitions than requested.')
                                to_do=max(0,to_do-processed)

                            for _ in range(to_do):   
                                results = {'Effort':[effort(N,n)],
                                           'Model':[model],
#                                            'representation':[representation],
                                           'Ranking Function':[rank_function],
                                           'Accuracy':[np.random.rand()],
                                           'Precision':[np.random.rand()],
                                           'Recall':[np.random.rand()],
                                           'F1-Score':[np.random.rand()],
                                          }
                                new_results=pd.DataFrame(results)
                                new_results['representation']=representation
                                new_results['category']=category

                                if os.path.isfile(results_file):
                                    old_results = pd.read_csv(results_file)
                                    new_results = old_results.append(new_results)
                                new_results.to_csv(results_file, index=False)
                            total+=to_do
#                             break
                
                
    info(f'Total TO-DO: {total}')
                        
#                             unlabeled = Dataset20NG.get_20newsgroup_unlabeled_collection()
#                             relevants = [item for item in unlabeled if oracle[item.id_]==DataItem20NG.RELEVANT_LABEL]

#                             rng = np.random.default_rng(2022)
#                             seed=int(rng.random()*10e6)
#                             labeled = list(rng.choice(relevants, size=1))
                            
#                             results = SCAL20NG(session_name='some session',
#                                                labeled_collection=labeled,
#                                                unlabeled_collection=unlabeled,
#                                                batch_size_cap=n,
#                                                random_sample_size=N,
#                                                target_recall=target_recall,
#                                                ranking_function=rank_function,
#                                                item_representation=representations,
#                                                oracle=oracle,
#                                                model_type=model,
#                                                seed=seed).run()

#                             new_results=pd.DataFrame(results)
#                             new_results['representation']=args.representation

#                             if os.path.isfile(results_file):
#                                 old_results = pd.read_csv(results_file)
#                                 new_results = old_results.append(new_results)
#                             new_results.to_csv(results_file, index=False)
                            
#                         done+=np.sum(mask)
#                         assert np.sum(mask)<no_of_seeds, np.sum(mask)
#                         remaining += no_of_seeds - np.sum(mask)
    
#     if os.path.isfile(output_path):
#         done=0
#         remaining=0 #len(Ns)*len(ns)*len(models)*len(representations)*len(sampling_functions)*no_of_seeds
#         df = pd.read_csv(output_path)


       

#         results = SCAL20NG(session_name='some session',
#                            labeled_collection=labeled,
#                            unlabeled_collection=unlabeled,
#                            batch_size_cap=args.n,
#                            random_sample_size=args.N,
#                            target_recall=args.target_recall,
#                            ranking_function=args.ranking_function,
#                            item_representation=representations,
#                            oracle=oracle,
#                            model_type=args.model_type,
#                            seed=args.seed).run()

#         new_results=pd.DataFrame(results)
#         new_results['category']=args.category
#         new_results['representation']=args.representation

#         if os.path.isfile(results_file):
#             old_results = pd.read_csv(results_file)
#             new_results = old_results.append(new_results)


#     new_results.to_csv(results_file, index=False)
    print('Hello world')