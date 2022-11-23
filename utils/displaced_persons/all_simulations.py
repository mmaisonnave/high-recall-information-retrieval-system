import os
import pandas as pd
import pandas
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('/home/ec2-user/SageMaker/mariano/repositories/tdmstudio-high-recall-information-retrieval-system/')

from utils.displaced_persons.scal import SCALDP
from utils.displaced_persons.dataset import DatasetDP, DataItemDP

from utils.io import info, warning


def how_many_processed(df, N, n, model, representation, rank_function):
    mask = (df['N']==N) & (df['n']==n) & (df['representation']==representation) \
                                & (df['Model']==model) & (df['Ranking Function']==rank_function)
    return np.sum(mask)

if __name__=='__main__':
    target_recall=0.8
     # CHANGE <<<<<<<<<
    dataset = 'displaced persons'
    dataset_size=7282
    no_of_seeds = 5
    Ns = [int(p*dataset_size) for p in [0.05, 0.10, 0.20, 0.25, 0.5, 0.75, 1.0]]
    ns = [1, 3, 5, 10, 20]
    models = ['logreg', 'svm']
    representations = ['bow', 'sbert', 'glove']
    sampling_functions = ['relevance',]
    
    results_file = '/home/ec2-user/SageMaker/mariano/datasets/displaced_persons/simulation_results/all_results_final.csv'
    df=None
    if os.path.isfile(results_file):
        df = pd.read_csv(results_file)
        info(f'Read previous results. Results found: {len(df):,}')
    
    total=0
    info(f'Total of repetitions requested: {len(representations)*len(models)*len(Ns)*len(ns)*no_of_seeds:,}')
    oracle = DatasetDP.get_DP_oracle()
    for representation in representations:
        representations = DatasetDP.get_DP_representations(type_=representation)
        for model in models:
            for rank_function in sampling_functions:
                for N in Ns:
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
                            unlabeled = DatasetDP.get_DP_unlabeled_collection()
                            assert len(unlabeled)==7282
                            for item in unlabeled:
                                item.set_unknown()
                            relevants = [item for item in unlabeled if oracle[item.id_]==DataItemDP.RELEVANT_LABEL]

                            rng = np.random.default_rng(2022)
                            seed=int(rng.random()*10e6)
                            labeled = list(rng.choice(relevants, size=1))
                            for item in labeled:
                                item.set_relevant()
                            labeled_ids={item.id_ for iten in labeled}
                            unlabeled = [item for item in unlabeled if not item.id_ in labeled_ids]

                            results = SCALDP(session_name='some session',
                                             labeled_collection=labeled,
                                             unlabeled_collection=unlabeled,
                                             batch_size_cap=n,
                                             random_sample_size=N,
                                             target_recall=target_recall,
                                             ranking_function=rank_function,
                                             item_representation=representations,
                                             oracle=oracle,
                                             model_type=model,
                                             seed=seed).run()

                            new_results=pd.DataFrame(results)
                            new_results['representation']=representation

                            if os.path.isfile(results_file):
                                old_results = pd.read_csv(results_file)
                                new_results = old_results.append(new_results)
                            new_results.to_csv(results_file, index=False)

            