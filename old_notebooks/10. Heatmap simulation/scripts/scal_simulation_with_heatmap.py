import sys
sys.path.append('/home/ec2-user/SageMaker/mariano/repositories/tdmstudio-high-recall-information-retrieval-system/')
from utils.oracle import Oracle
from utils.io import info
from utils.data_item import GenericSyntheticDocument
from utils.scal import SCAL
import spacy
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SCAL simulation')

    parser.add_argument('--N', dest='N', type=int, help='Size of random sample', required=True)
    parser.add_argument('--n', dest='n', type=int, help='Batch size cap', required=True)
    parser.add_argument('--seed', dest='seed', type=int, help='Seed number', required=True)
    parser.add_argument('--target-recall', dest='target_recall', type=float, help='Target recall for the SCAL process', required=True)
    
    choices=['relevance', 'relevance_with_avg_diversity', 'relevance_with_min_diversity', 'half_relevance_half_uncertainty', 
             '1quarter_relevance_3quarters_uncertainty', 'random', '3quarter_relevance_1quarters_uncertainty', 'avg_distance', 
             'uncertainty', 'min_distance', 'uncertainty_with_avg_diversity', 'uncertainty_with_min_diversity']
            
    parser.add_argument('--ranking-function', dest='ranking_function', type=str, 
                        help='Ranking function.', choices=choices, required=True)
    parser.add_argument('--representation', dest='representation', type=str, 
                        help='Vector representation.', required=True, choices=['GloVe','sbert','BoW'])
    
    args = parser.parse_args()

    # Unlabeled collection
    unlabeled = Oracle.get_collection()
    info(f'Number of relevant articles found: {len(unlabeled):,}')
    representations = Oracle.get_document_representation(type_=args.representation) 
    info(f'Number of representations found:   {len(representations):,} (shape={representations[list(representations)[0]].shape})')

    # Synthetic Document
    topic_description='displaced persons dp'

    doc = GenericSyntheticDocument(topic_description,
                                   vocab_path='/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_vocab/vocab.txt', 
                                   idf_path='/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_vocab/idf.txt',
                                  )
    doc.set_relevant()

    if args.representation=='BoW':
        representations[doc.id_] = doc.vector()
    elif args.representation=='GloVe':
        nlp = spacy.load('en_core_web_lg')
        representations[doc.id_] = nlp(topic_description).vector
    elif args.representation=='sbert':
        representations[doc.id_] = pickle.load(open('../sbert_synthetic_dp_vec.pickle', 'rb'))


    info(f'Topic description: {topic_description} (shape={doc.vector().shape})')


    session_name=f'rf_{args.ranking_function}_N_{args.N}_n_{args.n}_tr_{args.target_recall}_d_DP_repr_{args.representation}_seed_{args.seed}'

    info(f'RUNNING: {session_name}')

    SCAL(session_name, 
         labeled_collection=[doc], 
         unlabeled_collection=unlabeled,
         random_sample_size=args.N,
         batch_size_cap=args.n,
         simulation=True,
         target_recall=args.target_recall,
         seed=args.seed,
         ranking_function=args.ranking_function,
         item_representation=representations,
        ).run()