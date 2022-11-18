import os
import pickle
from utils.data_item import DataItem
from utils.tdmstudio import get_title_and_text

import spacy 
class Oracle(object):
    distilbert_path = '/home/ec2-user/SageMaker/mariano/huggingface/pretrained/distilbert-base-uncased/'

    label_file = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/labeled_data_latest_08072022.csv'
    precomputed_folder = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_precomputed/'
    data_path = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_files/'
#     precomputed_folder = '/home/ec2-user/SageMaker/mariano/notebooks/04. Model of DP/precomputed/'
    id2label = dict([(line.split(';')[0] ,line.split(';')[1]) for line in open(label_file,'r').read().splitlines()[1:] ])
    
#     invalid_ids=['1284635697', '1313942850'] #Obtained with `_print_corrupted_files`
    
    def is_relevant(data_item):
        assert data_item.id_ in Oracle.id2label, 'the label for that item is not available.'
        assert Oracle.id2label[data_item.id_]=='R' or Oracle.id2label[data_item.id_]=='I'
        return Oracle.id2label[data_item.id_]=='R'
    
    def get_collection():
#         collection = [DataItem(id_) for id_ in Oracle.id2label if not id_ in Oracle.invalid_ids] 
        collection = [DataItem(id_) for id_ in Oracle.id2label] 
        return collection

#     def _print_corrupted_files():
#         collection = [DataItem(id_) for id_ in Oracle.id2label] 
#         def invalid(id_):
#             try:
#                 pickle.load(open(os.path.join(Oracle.precomputed_folder, id_+ '_glove.p'),'rb'))
#                 return False
#             except:
#                 return True
#         collection = list(filter(lambda item: invalid(item.id_), collection))
#         print('Invalid documents: '+','.join([item.id_+'_glove.p' for item in collection]))
        
    def get_document_representation(type_='BoW'):
        if type_=='BoW':
            representation = {}
            files = [os.path.join(Oracle.precomputed_folder,file) for file in os.listdir(Oracle.precomputed_folder)]
            ids = [file.split('/')[-1][:-len('_representations.p')] for file in files]
            data = zip(ids, files)
            data = list(filter(lambda x: x[0] in Oracle.id2label, data))
            assert all([id_ in Oracle.id2label for id_,_ in data])

            for id_,file in data:
                representation[id_] = pickle.load(open(file, 'rb'))['BoW']

            return representation
        elif type_=='GloVe':
            if not os.path.isfile('glove_item_representation.pickle'):
                representation = {}
                files = [os.path.join(Oracle.data_path,file) for file in os.listdir(Oracle.data_path)]
                nlp = spacy.load('en_core_web_lg', disable=['textcat', 'parser', 'ner', 'tager', 'lemmatizer'])
                vectors = map(lambda file: nlp(get_title_and_text(file)).vector, files)

                for file, vector in zip(files,vectors):
                    id_= file.split('/')[-1][:-len('.xml')]
                    representation[id_]=vector

                pickle.dump(representation, open('glove_item_representation.pickle', 'wb'))
            else:
                representation = pickle.load(open('glove_item_representation.pickle', 'rb'))
            return representation
        elif type_=='sbert':
            if not os.path.isfile('sbert_item_representation.pickle'):
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(Oracle.distilbert_path)
                
                representation = {}
                
                files = [os.path.join(Oracle.data_path,file) for file in os.listdir(Oracle.data_path)]

#                 sentence_embeddings = model.encode(sentences)
                
                vectors =  model.encode(list(map(lambda file: get_title_and_text(file), files)))

                for file, vector in zip(files,vectors):
                    id_= file.split('/')[-1][:-len('.xml')]
                    representation[id_]=vector
                    
                pickle.dump(representation, open('sbert_item_representation.pickle', 'wb'))
            else:
#                 print('DEBUG::Reading sbert representation from pickle..', end='')
                representation = pickle.load(open('sbert_item_representation.pickle', 'rb'))
#                 print('[OK]')
            return representation

