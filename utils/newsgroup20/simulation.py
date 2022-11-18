import argparse
import sys 
import os
import pandas as pd
sys.path.append('/home/ec2-user/SageMaker/mariano/repositories/tdmstudio-high-recall-information-retrieval-system/')
from utils.newsgroup20.dataset import Dataset20NG, DataItem20NG


from utils.newsgroup20.scal import SCAL20NG
import numpy as np


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='20newsgroup SCAL simulation')

    parser.add_argument('--N', dest='N', type=int, help='Size of random sample', required=True)
    parser.add_argument('--category', dest='category', type=str, help='One of the 20 categories of the 20NG dataset.', required=True)
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
    


    representations = Dataset20NG.get_20newsgroup_representations(type_=args.representation) # CHANGE <<<<<<<<<
    

    unlabeled = Dataset20NG.get_20newsgroup_unlabeled_collection()
    oracle = Dataset20NG.get_20newsgroup_oracle(category=args.category)

    relevants = [item for item in unlabeled if oracle[item.id_]==DataItem20NG.RELEVANT_LABEL]

    rng = np.random.default_rng(2022)
    labeled = list(rng.choice(relevants, size=1))
    for item in labeled:
        item.set_relevant()

    labeled_ids = {item.id_ for item in labeled}
    unlabeled = [item for item in unlabeled if not item.id_ in labeled_ids]

    results_file = '/home/ec2-user/SageMaker/mariano/datasets/20news-18828/simulation_results/all_results.csv'


    results = SCAL20NG(session_name='some session',
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
    new_results['category']=args.category
    new_results['representation']=args.representation

    if os.path.isfile(results_file):
        old_results = pd.read_csv(results_file)
        new_results = old_results.append(new_results)


    new_results.to_csv(results_file, index=False)