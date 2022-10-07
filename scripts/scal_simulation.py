import sys
sys.path.append('..')

from utils.scal import SCAL
from utils.oracle import Oracle
from utils.data_item import QueryDataItem, GenericSyntheticDocument

import spacy
nlp = spacy.load('en_core_web_lg')
import re 

if __name__=='__main__':
    input_ = ' '.join(sys.argv)
    if len(re.findall('.*--N=([0-9][0-9]*).*', input_))==0:
        print('Please provide N (python script.py --N=50).')
        sys.exit(1)

    if len(re.findall('.*--target-recall=([0-9\.][0-9\.]*).*', input_))==0:
        print('Please provide target_recall (python script.py --target-recall=0.8).')
        sys.exit(1)
        
    if len(re.findall('.*--seed=([0-9][0-9]*).*', input_))==0:
        print('Please provide seed (python script.py --seed=123).')
        sys.exit(1)

    if len(re.findall('.*--proportion-relevance=([0-9][0-9]*).*', input_))==0:
        print('Please provide seed (python script.py --proportion-relevance=1.0).')
        sys.exit(1)


    if len(re.findall('.*--ranking-function=([a-z\_0-9]*).*', input_))==0:
        print('Please provide ranking function (python script.py --ranking-function=relevance_with_avg_diversity).')
        sys.exit(1)

    glove=False
    if re.search('--glove', input_):
        glove=True
        
    relevance_function = re.findall('.*--ranking-function=([0-9a-z\_]*).*', input_)[0]
    N = int(re.findall('.*--N=([0-9][0-9]*).*', input_)[0])
    target_recall=float(re.findall('.*--target-recall=([0-9\.][0-9\.]*).*', input_)[0])
    seed=int(re.findall('.*--seed=([0-9\.][0-9\.]*).*', input_)[0])
    proportion_relevance=float(re.findall('.*--proportion-relevance=([0-9\.][0-9\.]*).*', input_)[0])
#     diversity = not re.search('.*--diversity', input_) is None
#     average_diversity = not re.search('.*--average-diversity', input_) is None
    session_name=f'simulation_tr_{int(target_recall*100)}_N_{N}_proportion_{int(proportion_relevance*100)}_{relevance_function}_seed_{seed}'
    if glove:
        session_name=session_name+'_glove'
#     if diversity:
#         session_name = session_name + '_diversity'
#     if average_diversity:
#         session_name = session_name + '_avg'

    unlabeled = Oracle.get_collection()
    if glove:
        representations = Oracle.get_document_representation(type_='GloVe')
    else:
        representations = Oracle.get_document_representation(type_='BoW')

    print(f'starting session_name: {session_name}')
    doc = GenericSyntheticDocument('dp',
                                   vocab_path='/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_vocab/vocab.txt', 
                                   idf_path='/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_vocab/idf.txt',
#                                    vocab_path='/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_vocab/vocab.txt', 
#                                    idf_path='/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_vocab/idf.txt',
                                  )
    doc.set_relevant()

    if glove:
        representations[doc.id_] = nlp('dp').vector
    else:
        representations[doc.id_] = doc.vector()
    
    
    assert len(unlabeled)+len([doc])==len(representations)

    SCAL(session_name, 
         [doc], 
         unlabeled,
         random_sample_size=N,
         simulation=True,
         target_recall=target_recall,
         proportion_relevance_feedback=proportion_relevance,
         seed=seed,
         ranking_function=relevance_function,
         item_representation=representations,
#          diversity=diversity,
#          average_diversity=average_diversity,
        ).run()
