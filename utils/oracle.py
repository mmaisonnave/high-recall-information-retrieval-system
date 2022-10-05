import os
import pickle
from utils.data_item import DataItem

class Oracle(object):
    label_file = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/labeled_data_latest_08072022.csv'
    precomputed_folder = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/data/DP_precomputed/'
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
        representation = {}
        files = [os.path.join(Oracle.precomputed_folder,file) for file in os.listdir(Oracle.precomputed_folder)]
        ids = [file.split('/')[-1][:-len('_representations.p')] for file in files]
        data = zip(ids, files)
        data = list(filter(lambda x: x[0] in Oracle.id2label, data))
        assert all([id_ in Oracle.id2label for id_,_ in data])
        
        for id_,file in data:
            representation[id_] = pickle.load(open(file, 'rb'))['BoW']
            
        return representation
    
