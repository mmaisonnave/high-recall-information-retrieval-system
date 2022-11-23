import argparse
import sys 
import os
import pandas as pd
sys.path.append('/home/ec2-user/SageMaker/mariano/repositories/tdmstudio-high-recall-information-retrieval-system/')
from utils.displaced_persons.dataset import DatasetDP, DataItemDP


from utils.displaced_persons.scal import SCALDP
import numpy as np


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DP SCAL simulation')

    parser.add_argument('--N', dest='N', type=int, help='Size of random sample', required=True)
    parser.add_argument('--n', dest='n', type=int, help='Batch size cap', required=True)
    parser.add_argument('--seed', dest='seed', type=int, help='Seed number', required=True)
    parser.add_argument('--target-recall', dest='target_recall', type=float, help='Target recall for the SCAL process', required=True)
    parser.add_argument('--model-type', dest='model_type', type=str, help='Model type', choices=['svm', 'logreg'], required=True)
    parser.add_argument('--representation',dest='representation', type=str, help='vectors',choices=['bow', 'sbert', 'glove'],required=True)
    
    choices=['relevance', ]
#              'relevance_with_avg_diversity', 'relevance_with_min_diversity', 'half_relevance_half_uncertainty', 
#              '1quarter_relevance_3quarters_uncertainty', 'random', '3quarter_relevance_1quarters_uncertainty', 'avg_distance', 
#              'uncertainty', 'min_distance', 'uncertainty_with_avg_diversity', 'uncertainty_with_min_diversity']
            
    parser.add_argument('--ranking-function', dest='ranking_function', type=str, 
                        help='Ranking function.', choices=choices, required=True)
    
    args = parser.parse_args()
    


    representations = DatasetDP.get_DP_representations(type_=args.representation) # CHANGE <<<<<<<<<
    

    unlabeled = DatasetDP.get_DP_unlabeled_collection()
    oracle = DatasetDP.get_DP_oracle()

    relevants = [item for item in unlabeled if oracle[item.id_]==DataItemDP.RELEVANT_LABEL]

    rng = np.random.default_rng(2022)
    labeled = list(rng.choice(relevants, size=1))
    for item in labeled:
        item.set_relevant()

    labeled_ids = {item.id_ for item in labeled}
    unlabeled = [item for item in unlabeled if not item.id_ in labeled_ids]

    results_file = '/home/ec2-user/SageMaker/mariano/datasets/displaced_persons/simulation_results/all_results.csv'


    results = SCALDP(session_name='some session',
                       labeled_collection=labeled,
                       unlabeled_collection=unlabeled,
                       batch_size_cap=args.n,
                       random_sample_size=args.N,
                       target_recall=args.target_recall,
                       ranking_function=args.ranking_function,
                       item_representation=representations,
                       oracle=oracle,
                       model_type=args.model_type,
                       seed=args.seed).run()
    
    new_results=pd.DataFrame(results)
    new_results['representation']=args.representation

    if os.path.isfile(results_file):
        old_results = pd.read_csv(results_file)
        new_results = old_results.append(new_results)


    new_results.to_csv(results_file, index=False)