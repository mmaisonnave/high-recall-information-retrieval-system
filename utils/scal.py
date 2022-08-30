import numpy as np
from utils.term_highlighter import TermHighlighter
from utils.data_item import DataItem,QueryDataItem
import threading
import myversions.pigeonXT as pixt
from utils.io import html
from threading import Thread
import time
import logging
import pandas as pd
from IPython.display import clear_output
from utils.oracle import Oracle
import json
import os
import logging

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

class SCAL(object):    
    RELEVANT_LABEL='Relevant'
    IRRELEVANT_LABEL='Irrelevant'
    def __init__(self,
                 session_name=None,
                 labeled_collection=None,
                 unlabeled_collection=None,
                 batch_size_cap=10,
                 random_sample_size=10000,
                 target_recall=0.8,
                 simulation=False,
                 empty=False,
                 seed=2022
                ):
        
        if not empty:
            if not os.path.exists(f'sessions/scal/{session_name}/log/'):
                if not os.path.exists(f'sessions/scal/{session_name}/'):
                    os.mkdir(f'sessions/scal/{session_name}/')
                os.mkdir(f'sessions/scal/{session_name}/log/')
            logging.basicConfig(filename=f'sessions/scal/{session_name}/log/scal_system.log', 
                                format='%(asctime)s [%(levelname)s] %(message)s' ,
    #                             encoding='utf-8',                # INVALID WHEN CHANGE ENV (IMM -> BERT)
                                datefmt='%Y-%m-%d %H:%M:%S',
    #                             force=True,                      # INVALID WHEN CHANGE ENV (IMM -> BERT)
                                level=logging.DEBUG)
            logging.debug(f'STARTING SCAL - Size of initial labeled={len(labeled_collection)} - '\
                          f'Size of full unlabeled collection={len(unlabeled_collection)} - '\
                          f'Size of Random Sample={random_sample_size} - '\
                          f'Batch size cap={batch_size_cap} - '\
                          f'Target recall={target_recall} - Simualation={simulation} - Session Name={session_name}')
            self.simulation=simulation
            self.session_name=session_name
            self.target_recall=target_recall
            self.random_sample_size=random_sample_size
            self.seed = seed
            self.ran = np.random.default_rng(self.seed)
            self.B = 1 
            self.n = batch_size_cap
            self.random_unlabeled_collection = self.ran.choice(unlabeled_collection, 
                                                          size=min(self.random_sample_size,len(unlabeled_collection)), 
                                                          replace=False)
            self.U0 = self.random_unlabeled_collection
            self.cant_iterations = SCAL._cant_iterations(len(self.random_unlabeled_collection))
            self.Rhat=np.zeros(shape=(self.cant_iterations+1,))
            self.it=1
            self.labeled_collection = labeled_collection
            self.unlabeled_collection = unlabeled_collection
            self.removed = []
            self.models=[]
            self.precision_estimates=[]
            assert all([item.is_relevant() or item.is_irrelevant() for item in labeled_collection])
            self.all_texts = [item.get_htmldocview() for item in labeled_collection]
            self.all_labels = [SCAL.RELEVANT_LABEL if item.is_relevant() else SCAL.IRRELEVANT_LABEL for item in labeled_collection]
        
    def to_disk(self):
        configuration = {'session-name':self.session_name,
                         'target-recall':self.target_recall,
                         'seed':self.seed,
                         'simulation':self.simulation,
                         'random-sample-size':self.random_sample_size,
                         'B': self.B,
                         'n': self.n,
                         'random-unlabeled-collection': [item._dict() for item in self.random_unlabeled_collection],
                         'U0': [item._dict() for item in self.U0], 
                         'cant-iterations': self.cant_iterations,
                         'Rhat': list(self.Rhat),
                         'it': self.it,
                         'labeled-collection': [item._dict() for item in self.labeled_collection],
                         'unlabeled-collection': [item._dict() for item in self.unlabeled_collection],
                         'removed': [[elem._dict() for elem in list_] for list_ in self.removed],
#                          'models': self.models,
                         'precision-estimates': self.precision_estimates,
                         'all-texts': self.all_texts,
                         'all-labels': self.all_labels,
                        }

        if not os.path.exists(f'sessions/scal/{self.session_name}/data/'):
            
#             (f'sessions/scal/{self.session_name}')
            os.mkdir(f'sessions/scal/{self.session_name}/data')
#             if not os.path.exists(f'sessions/scal/{session_name}/log/'):
#                 os.mkdir(f'sessions/scal/{self.session_name}/log')
            os.mkdir(f'sessions/scal/{self.session_name}/models')
            
        for idx,model in enumerate(self.models):
            model.to_disk(f'sessions/scal/{self.session_name}/model_{idx}')
        with open(f'sessions/scal/{self.session_name}/data/configuration.json','w') as outputfile:
            outputfile.write(json.dumps(configuration, indent=4)) 
                
        
    def from_disk(session_name):
        with open(f'sessions/scal/{session_name}/data/configuration.json', 'r') as f:
            configuration = json.load(f)
        scal = SCAL(empty=True)
        scal.session_name=configuration['session-name'] #session_name
        scal.simulation=configuration['simulation'] #session_name
        scal.target_recall=configuration['target-recall']
        scal.random_sample_size= configuration['random-sample-size']
        self.seed=configuration['seed']
        scal.ran =  np.random.default_rng(self.seed)
        scal.B = configuration['B']
        scal.n = configuration['n']
        scal.random_unlabeled_collection = [DataItem.from_dict(dict_) for dict_ in configuration['random-unlabeled-collection']] 
        scal.U0 = [DataItem.from_dict(dict_) for dict_ in configuration['U0']] 
        scal.cant_iterations =  configuration['cant-iterations']
        scal.Rhat= np.array(configuration['Rhat']) #np.zeros(shape=(self.cant_iterations+1,))
        scal.it=configuration['it']            
        scal.labeled_collection = [DataItem.from_dict(dict_) if 'id' in dict_ else QueryDataItem.from_dict(dict_)  
                                   for dict_ in configuration['labeled-collection']] 
        
        scal.unlabeled_collection = [DataItem.from_dict(dict_) for dict_ in configuration['unlabeled-collection']] 

        scal.removed = [[DataItem.from_dict(dict_) for dict_ in list_] for list_ in configuration['removed']]
#         scal.models=configuration['models']
        scal.precision_estimates=configuration['precision-estimates']
    
        scal.models=[]
        for idx,model in enumerate([file for file in os.listdir(f'sessions/scal/{session_name}/') if 'model_' in file and file.endswith('json')]):
            scal.models.append(TermHighlighter.from_disk(f'sessions/scal/{session_name}/{model[:-5]}'))
            
        scal.all_texts = configuration['all-texts']
        scal.all_labels = configuration['all-labels']
        logging.basicConfig(filename=f'sessions/scal/{session_name}/log/scal_system.log', 
                            format='%(asctime)s [%(levelname)s] %(message)s' ,
#                             encoding='utf-8',                # INVALID WHEN CHANGE ENV (IMM -> BERT)
                            datefmt='%Y-%m-%d %H:%M:%S',
#                             force=True,                      # INVALID WHEN CHANGE ENV (IMM -> BERT)
                            level=logging.DEBUG)

        return scal
        # use from_dict in DataItem and change Rhat to numpy
    def resume(self):
        pass
    def run(self):
        self.loop()
    def _extend_with_random_documents(self):        
        extension = self.ran.choice(self.random_unlabeled_collection, size=min(100,len(self.random_unlabeled_collection)), replace=False)
        list(map(lambda x: x.set_irrelevant(), self.random_unlabeled_collection))
        return extension
    
    def _label_as_unkown(collection):
        list(map(lambda x: x.set_unknown(), collection))
        
    def _build_classifier(training_collection):
        model = TermHighlighter()
        model.fit(training_collection)
        return model
    def _select_highest_scoring_docs(self):
        yhat = self.models[-1].predict(self.random_unlabeled_collection)
        args = np.argsort(yhat)[::-1]
        return [self.random_unlabeled_collection[arg] for arg in args[:self.B]]
        
    def _remove_from_unlabeled(self,to_remove):
        to_remove = set(to_remove)
        return list(filter(lambda x: not x in to_remove, self.random_unlabeled_collection))

    
             
    def _total_effort(self):    
        B=1
        it=1
        effort=0
        len_unlabeled=self._unlabeled_in_sample()
        while (B<len_unlabeled):        
            b = B if B<=self.n else self.n
            effort+=b
            len_unlabeled = len_unlabeled - B
            B+=int(np.ceil(B/10))
            it+=1
        return effort   
    def _cant_iterations(len_unlabeled):    
        B=1
        it=1
        while (B<len_unlabeled):        
            len_unlabeled = len_unlabeled - B
            B+=int(np.ceil(B/10))
            it+=1
        return it
    
    def _relevant_count(self):
        return len([item for item in self.labeled_collection if item.is_relevant()])
    def _irrelevant_count(self):
        return len([item for item in self.labeled_collection if item.is_irrelevant()])
    def _unlabeled_in_sample(self):
        return len(self.random_unlabeled_collection)
    
    def _progress_bar(self,size):
        effort = len(self.labeled_collection)
        total_effort = self._total_effort()+1 # +1 for the topic description (which is included in the labeled collection)
        str_=f'{int(100*(effort/total_effort)):3} %'
        # print(f'{int(effort/total_effort):3}', end='')\
        str_+=' |'
        end_=f'| {effort:4}/{total_effort:4}'
        str_+=int((size-(len(str_)+len(end_)))*(effort/total_effort))*'='
        str_+= '-' if (effort%total_effort)!=0 else ''
        print(str_+' '*(size-len(str_)-len(end_)) +end_)
        
    def loop(self):
        # STATUS BAR
        if not self.simulation:
            print('-'*109)
            print(f'Session name: {self.session_name}')
            print(f'Labeled documents: {len(self.labeled_collection)} '\
                  f'({self._relevant_count():8,} relevant / {self._irrelevant_count():8,} irrelevants)\t\t'\
                  f'Unlabeled documents: {self._unlabeled_in_sample():8,}')
            self._progress_bar(109)
            print('-'*109)
            # ~
        
        
        self.b = self.B if (self.Rhat[self.it-1]==1 or self.B<=self.n) else self.n
        assert self.b<=self.B
        precision = f'{self.precision_estimates[-1]:4.3f}' if len(self.precision_estimates)>0 else 'N/A'
        
        logging.debug(f'it={self.it:>4}/{self.cant_iterations:4} - B={self.B:<5,} - b={self.b:2} - Rhat={self.Rhat[self.it-1]:8.3f} -'\
              f' len_unlabeled= {len(self.random_unlabeled_collection):6,} - len_labeled={len(self.labeled_collection):6,}'\
              f' - cant_rel={self._relevant_count()} - precision={precision}'
             ) 
        
        extension = self._extend_with_random_documents()
        

        
        self.models.append(SCAL._build_classifier(list(extension)+list(self.labeled_collection)))
        SCAL._label_as_unkown(extension)
        self.sorted_docs = self._select_highest_scoring_docs()

        self.random_sample_from_batch = self.ran.choice(self.sorted_docs, size=self.b, replace=False)

        yhat = self.models[-1].predict(self.random_sample_from_batch)
        text_for_label = [suggestion.get_htmldocview(highlighter=self.models[-1], confidence_score=confidence)
                          for suggestion,confidence in zip(self.random_sample_from_batch,yhat)]
        client_current_index = len(self.all_texts)+1
        self.all_texts += text_for_label
        df = pd.DataFrame(
                       {
                        'example': self.all_texts,
                        'changed':[True]*len(self.all_texts),
                        'label':self.all_labels+([None]*len(text_for_label))
                       }
                      )
        if not self.simulation:
            self.annotations = pixt.annotate(#text_for_label,
                                             df,
                                             options=[SCAL.RELEVANT_LABEL, SCAL.IRRELEVANT_LABEL],
                                             stop_at_last_example=False,
                                             display_fn=html,
                                             cancel_process_fn=None,
                                             final_process_fn=self.after_loop,
                                             client_current_index=client_current_index,
                                             finish_button_label='save & next batch',
                                             include_cancel=False,
                                            )
        else:
            self.after_loop()
        
    def after_loop(self):
        if not self.simulation:
            self.all_labels = list(self.annotations["label"])
        else:
            # SIMULATION
            self.all_labels += [ SCAL.RELEVANT_LABEL if Oracle.is_relevant(item) else  SCAL.IRRELEVANT_LABEL 
                                  for item in self.random_sample_from_batch ] 
            
        self.labeled_collection = list(self.labeled_collection) + list(self.random_sample_from_batch)
        for item,label in zip(self.labeled_collection, self.all_labels):
            assert label==SCAL.RELEVANT_LABEL or label==SCAL.IRRELEVANT_LABEL
            label = DataItem.REL_LABEL if label==SCAL.RELEVANT_LABEL else DataItem.IREL_LABEL
            item.assign_label(label)   
#         print(f'cantidad de relevantes={len([item for item in self.labeled_collection if item.is_relevant()])}')     

            
        self.random_unlabeled_collection = self._remove_from_unlabeled(self.sorted_docs)    
        self.removed.append([elem for elem in self.sorted_docs ])


        r = len([item for item in self.random_sample_from_batch if item.is_relevant()])
        self.precision_estimates.append(r/self.b)
        self.Rhat[self.it] = (r*self.B)/self.b
        if self.it>1:
            self.Rhat[self.it] += self.Rhat[self.it-1]

        self.B += int(np.ceil(self.B/10))
        self.B = min(self.B, len(self.random_unlabeled_collection))

        self.it+=1
        self.to_disk()

        
        if len(self.random_unlabeled_collection)>0:
            if not self.simulation:
                clear_output(wait=False)
            self.loop()
        else:
            self.finish()


    def finish(self):
        logging.debug(f'it=  end    - B={self.B:<5,} - b={self.b:2} - Rhat={self.Rhat[self.it-1]:8.3f} -'\
              f' len_unlabeled= {len(self.random_unlabeled_collection):6,} - len_labeled={len(self.labeled_collection):6,}'\
              f' - cant_rel={self._relevant_count()} - precision={self.precision_estimates[-1]:4.3f}'
             )

        self.prevalecence = (1.05*self.Rhat[self.it-1]) / self.random_sample_size
#         print(f'Prevalecense: {self.prevalecence}')
        
        no_of_expected_relevant = self.target_recall * self.prevalecence * self.random_sample_size

        j=0
        while j<len(self.Rhat) and self.Rhat[j]<no_of_expected_relevant:
            j+=1
            

            
#         print(f'j value: {j} (it={j+1})')
        Uj = self.U0

        to_remove = set([elem for list_ in self.removed[:j]  for elem in list_])
#         print(f'size of removed: {len(self.removed)}')
#         print(f'size of to_remove: {len(to_remove)}')
        Uj = [elem for elem in Uj if not elem in to_remove]
#         print(f'Size of Uj: {len(Uj)}')
        
        t = np.max(self.models[j].predict(Uj))
#         print(f'threshold={t}')
        self.models.append(SCAL._build_classifier(self.labeled_collection))
        
        print(f'Size of unlabeled={len(self.unlabeled_collection)}')
        print(f'Size of labeled={len(self.labeled_collection)}')
        final_unlabeled_collection = [item for item in self.unlabeled_collection if not item in self.labeled_collection[1:]]
        print(f'Size of unlabeled after removing labeled={len(final_unlabeled_collection)}')
        
        yhat = self.models[-1].predict(final_unlabeled_collection)
        relevant = yhat>=t
        print(f'Shape of yhat={yhat.shape}')
        print(f'Number of relevant={np.sum(yhat>=t)}')
        
        logging.debug(f'SCAL RESULTS: Est. prevalecence={self.prevalecence:4.3f} - '\
                      f'Size of labeled={len(self.labeled_collection)} - '\
                      f'Size of total unlabeled={len(final_unlabeled_collection)} - '\
                      f'Est. relevant in sample={no_of_expected_relevant:4.3f} - '\
                      f'It used for determine threshold={j+1} (j={j}) - Threshold={t:4.3f} - '\
                      f'Relevant found (total): {len([item for item in self.labeled_collection if item.is_relevant()])+np.sum(relevant)}'\
                      f'({len([item for item in self.labeled_collection if item.is_relevant()])} labeled / {np.sum(relevant)} suggested)'
                     )
#         print(f'Relevant count: {np.sum(relevant)}')
#         print(f'Precision estimate: {np.average(self.precision_estimates)}')
        
        labeled_data = [item for item in self.labeled_collection[1:] if item.is_relevant()]
        confidence = [1.0]*len(labeled_data)
        
        no_of_labeled_rel = len(labeled_data)
        
        labeled_data += [item for item,y in zip(final_unlabeled_collection,yhat) if y>=t]
        confidence +=list([y for item,y in zip(final_unlabeled_collection,yhat) if y>=t])
        
        assert len(labeled_data)==len(confidence)
        
        print(f'len(labeled_data)={len(labeled_data)}')
        print(f'len(labeled_data)={len(confidence)}')
        filename = f'sessions/scal/{self.session_name}/data/exported_data_'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'
        with open(filename, 'w') as writer:
            writer.write('URL,relevant_or_suggested,confidence\n')
            count=0
            for item,confidence_value in zip(labeled_data,confidence):
                if count<no_of_labeled_rel:
                    writer.write(f'https://proquest.com/docview/{item.id_},rel,{confidence_value:4.3f}\n')  
                else:
                    writer.write(f'https://proquest.com/docview/{item.id_},sugg,{confidence_value:4.3f}\n')  
                count+=1
                
                      
#         if send_email:
#             assert os.path.isfile(filename)
#             temp_file ='exported_data_'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'
#             shutil.copyfile(filename, temp_file)
#             assert os.path.isfile(temp_file)
#             os.system(f'aws s3 cp {temp_file} s3://pq-tdm-studio-results/tdm-ale-data/623/results/')
#             os.remove(temp_file)
#             logging.debug('E-mail sending enabled, email sent [OK].')
            
        if self.simulation:
            ytrue = [1 if Oracle.is_relevant(item) else 0 for item in final_unlabeled_collection]
            acc = accuracy_score(ytrue,yhat>=t)
            prec = precision_score(ytrue, yhat>=t)
            rec = recall_score(ytrue, yhat>=t)
            f1 = f1_score(ytrue, yhat>=t)
            tn, fp, fn, tp = confusion_matrix(ytrue, yhat>=t).ravel()
            logging.debug(f'SIMULATION RESULTS: accuracy={acc:4.3f} - precision={prec:4.3f} - recall={rec:4.3f} - F1-score={f1:4.3f} - '\
                          f'TN={tn} - FP={fp} - FN={fn} - TP={tp} ')
#             print(f'Accuracy:  {acc:4.3f}')
#             print(f'Precision: {prec:4.3f}')
#             print(f'Recall:    {rec:4.3f}')
#             print(f'F1-score   {f1:4.3f}')
        print('FINISH simulation')
        return labeled_data, confidence
#         return _get_relevant(model, prevalecence, unlabeled_collection)
        
