from utils.io import warning, html, request_user_input, info
from utils.auxiliary import has_duplicated
import numpy as np
from utils.data_item import DataItem
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class Simulator(object):
    def __init__(self):
        self.suggestions = []
    def _simulated_loop(
        self, 
        models,
        arg_rank_function,
    ):
        self.simulation_results['loop_no'].append(self.iteration_count)
        info(f' ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ LOOP NO. {self.iteration_count} ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ')
        info(f'relevant count:   {self.get_relevant_count()}')
        info(f'irrelevant count: {self.get_irrelevant_count()}')
        self.simulation_results['relevant_count'].append(self.get_relevant_count())
        self.simulation_results['irrelevant_count'].append(self.get_irrelevant_count())
        assert len(self.suggestions)==0
        
        ids_list = [item.id_ for item in self.labeled_data]
        ids_list += [item.id_ for item in self.unlabeled_data]
        ids_list += [item.id_ for item in self.suggestions]
        assert not has_duplicated(ids_list)
        del(ids_list)
        
        # # # # # # # # # # # # # #
        # ASSESMENT ON TRAIN DATA #
        # # # # # # # # # # # # # #
#         y_true = np.array([self.id2label[item.id_] for item in self.unlabeled_data])==DataItem.REL_LABEL
        y_true = np.array([self.id2label[item.id_] for item in self.labeled_data])==DataItem.REL_LABEL
        yhat = np.zeros(shape=(len(self.labeled_data)))

        elapsed_time=0
        for idx, model in enumerate(models):
            print(f'computing on model: {str(model)}')
            start = time.time()
            yhat +=  model.predict(self.labeled_data)  ###  ALSO DO TESTING AND SAVE *BOTH*
            elapsed_time += time.time()-start 
        yhat = yhat/len(models)
        yhat = yhat>0.5
        
        self.simulation_results[f'train_accuracy'].append(accuracy_score(y_true, yhat))
        self.simulation_results[f'train_precision'].append(precision_score(y_true, yhat))
        self.simulation_results[f'train_recall'].append(recall_score(y_true, yhat))
        self.simulation_results[f'train_f1'].append(f1_score(y_true, yhat))

        
        # # # # # # # # # # # # # #
        # ASSESMENT ON TEST DATA  #
        # # # # # # # # # # # # # #
#         y_true = np.array([self.id2label[item.id_] for item in self.unlabeled_data])==DataItem.REL_LABEL
        y_true = np.array([item.label for item in self.test_data])==DataItem.REL_LABEL
        yhat = np.zeros(shape=(len(self.test_data)))

        
        for idx, model in enumerate(models):
            print(f'computing on model: {str(model)}')
            start = time.time()
            yhat +=  model.predict(self.test_data)  ###  ALSO DO TESTING AND SAVE *BOTH*
            elapsed_time += time.time()-start 

        yhat = yhat/len(models)
        yhat = yhat>0.5
        
        TP = np.sum(yhat & y_true)
        
        self.simulation_results[f'test_accuracy'].append(accuracy_score(y_true, yhat))
        self.simulation_results[f'test_precision'].append(precision_score(y_true, yhat))
        self.simulation_results[f'test_recall'].append(recall_score(y_true, yhat))
        self.simulation_results[f'test_f1'].append(f1_score(y_true, yhat))
        self.simulation_results[f'test_TP'].append(TP)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
        # HAVE TO COMPUTE THE YHAT ON THE UNLABELED (SO FAR I COMPUTED ON TESTING AND ON LABELED). #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #
        yhat = np.zeros(shape=(len(self.unlabeled_data)))
        y_true = np.array([self.id2label[item.id_] for item in self.unlabeled_data])==DataItem.REL_LABEL

        for idx, model in enumerate(models):
            print(f'computing on model: {str(model)}')
            start = time.time()
            yhat +=  model.predict(self.unlabeled_data)  ###  ALSO DO TESTING AND SAVE *BOTH*
            elapsed_time += time.time()-start 
        yhat = yhat/len(models)
        yhat = yhat>0.5
        
        
        TP = np.sum(yhat & y_true)
        
        self.simulation_results[f'unlabeled(train)_TP'].append(TP)
        
        total_instances =len(self.test_data)+len(self.labeled_data)+len(self.unlabeled_data)   
        self.simulation_results[f'prediction_time'].append(elapsed_time/(total_instances))
        # TAKE THE BEST TEN OUT OF UNLABELED AND MOVE TO LABELED WITH THE GROUND-TRUTH LABEL
        sorted_args = arg_rank_function(yhat) #list(reversed(np.argsort(yhat)))

                
        sorted_args = sorted_args[:self.labeling_batch_size]   
        
        new_batch = [self.unlabeled_data[arg] for arg in sorted_args]    # list index out of range
        for item in new_batch:
            if self.id2label[item.id_]==DataItem.REL_LABEL:
                item.set_relevant()
            else:
                assert self.id2label[item.id_]==DataItem.IREL_LABEL
                item.set_irrelevant()
        new_batch_id = set([item.id_ for item in new_batch])
        self.unlabeled_data = [item for item in self.unlabeled_data if not item.id_ in new_batch_id]
        self.labeled_data += new_batch 
        
        info('AFTER LABELING')
        info(f'relevant count:   {self.get_relevant_count()}')
        info(f'irrelevant count: {self.get_irrelevant_count()}')
        # RETRAIN
        elapsed_time=0
        for idx,model in enumerate(models):
            start = time.time()
            model.fit(self.labeled_data)
            elapsed_time += time.time()-start
            
        self.simulation_results[f'training_time'].append(elapsed_time/len(self.labeled_data))
        
        self.iteration_count+=1
 
        
    def simulation(
        self,
        train_data,
        test_data,
        arg_rank_function,
        models,
        starting_no_of_examples=5, 
        random_state=None, 
        labeling_batch_size=10,
    ):                
        self.test_data = test_data
        self.iteration_count=0
        self.simulation_results = {'loop_no': [],
                                   'relevant_count': [],
                                   'irrelevant_count': [],
                                   'prediction_time': [],
                                   'training_time': [],
                                   'train_accuracy': [],
                                   'train_precision': [],
                                   'train_recall': [],
                                   'train_f1': [],
                                   'test_accuracy': [],
                                   'test_precision': [],
                                   'test_recall': [],
                                   'test_f1': [],
                                   'test_TP': [],
                                   'unlabeled(train)_TP': [],
                                  }

       
                                   
                                   
        self.labeling_batch_size = labeling_batch_size
        # CHANGE LOGGING TO SIMULATION LOG FILE

        
                        
        if random_state is None:
            random_state = np.random.default_rng(2022)
        info(f'Number of labeled instances for training=   {len(train_data):12,}')
        info(f'Number of labeled instances for testing=   {len(test_data):12,}')
        # Saving labels
        self.id2label = dict([(item.id_, item.label) for item in train_data+test_data])
        
        # Rebuilding data lists (labeled only with few examples, everything else to unlabeled).
        relevant_args = [arg for arg in range(len(train_data)) if train_data[arg].is_relevant()]
        selected_args = set(random_state.choice(relevant_args, size=starting_no_of_examples, replace=False,))
        self.unlabeled_data = [train_data[arg] for arg in range(len(train_data)) if not arg in selected_args ]
        self.labeled_data = [train_data[arg] for arg in selected_args ]
        
        for item in self.unlabeled_data:
            item.set_unknown()
        
        info(f'len(unlabeled_data)= {len(self.unlabeled_data):12,}')
        info(f'len(labeled_data)=   {len(self.labeled_data):12,}')
        info(f'len(rel labeled_data)=   {len([item for item in self.labeled_data if item.is_relevant()]):12,}')
        assert len(set([item.id_ for item in self.unlabeled_data]).intersection(set([item.id_ for item in self.labeled_data])))==0
        
        
        

        #FIRST TRAIN WITH EXPANSION! THEN LOOP.
        expansion = [self.unlabeled_data[arg] for arg in random_state.choice(range(len(self.unlabeled_data)), size=10, replace=False,)]
        for item in expansion:
            item.set_irrelevant()
            
        
        for model in models:
            model.fit(self.labeled_data+expansion)
        for item in expansion:
            item.set_unknown()
        
        while len(self.unlabeled_data)>0:
            self._simulated_loop(models=models,
                                 arg_rank_function=arg_rank_function, 
                                 )         

        return self.simulation_results
    def get_relevant_count(self):
        return len([item for item in self.labeled_data if item.is_relevant()])
    def get_irrelevant_count(self):
        return len([item for item in self.labeled_data if  item.is_irrelevant()])