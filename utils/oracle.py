from utils.data_item import DataItem

class Oracle(object):
    label_file = '/home/ec2-user/SageMaker/mariano/notebooks/07. Simulation/labeled_data_latest_08072022.csv'
    id2label = dict([(line.split(';')[0] ,line.split(';')[1]) for line in open(label_file,'r').read().splitlines()[1:] ])
    
    def is_relevant(data_item):
        assert data_item.id_ in Oracle.id2label, 'the label for that item is not available.'
        assert Oracle.id2label[data_item.id_]=='R' or Oracle.id2label[data_item.id_]=='I'
        return Oracle.id2label[data_item.id_]=='R'
    
    def get_collection():
        return [DataItem(id_) for id_ in Oracle.id2label] 
        