from sklearn.metrics import confusion_matrix
import time
import os
from utils.data_item import DataItem
import pandas as pd
import numpy as np 
import myversions.pigeonXT as pixt
import pickle
from utils.io import warning, html, request_user_input, info 
import re
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from utils.classifier import Classifier
from utils.term_highlighter import TermHighlighter
from joblib import load,dump
import logging
import ipywidgets as widgets
from utils.auxiliary import has_duplicated  
import shutil 
from scipy import sparse
from sklearn.preprocessing import normalize
from utils.stopping_point import Quant_estimator,QuantCI_estimator
import threading

class HRSystem(object):
#     EXPANSION=10
    RELEVANT_LABEL='Relevant'
    IRRELEVANT_LABEL='Irrelevant'
    def __init__(self, from_scratch=False, session_name='default', finish_function=None, debug=False,print_function=None, ui=None):
        self.thread_count=0
        self.lock = threading.Lock()
        self.ui = ui
        self.loop_started=False
        if not os.path.exists(f'sessions/{session_name}'):
            logging.debug('Session directories do not exist, creating (data, log, models).')
            os.mkdir(f'sessions/{session_name}')
            os.mkdir(f'sessions/{session_name}/data/')
            os.mkdir(f'sessions/{session_name}/log/')
            os.mkdir(f'sessions/{session_name}/models/')
#         self.confidence_value = 0.5
        logging.basicConfig(filename=f'sessions/{session_name}/log/system.log', 
                            format='%(asctime)s [%(levelname)s] %(message)s' ,
#                             encoding='utf-8',                # INVALID WHEN CHANGE ENV (IMM -> BERT)
                            datefmt='%Y-%m-%d %H:%M:%S',
#                             force=True,                      # INVALID WHEN CHANGE ENV (IMM -> BERT)
                            level=logging.DEBUG)
        self.status=''
        logging.debug('#'*30+'  INIT  '+'#'*30)
        self.candidate_args = []
#         self.suggestion_cap = suggestion_cap
        self.session_name = session_name
        self.finish_function = finish_function
        self.debug=debug
        if not print_function is None:
            self.print_fn = print_function
        else:
            self.print_fn = print
            
        if not from_scratch and not finish_function is None:
            logging.debug('Not starting from scratch. However, finish function provided. It will be used during loop but not in init.')
        self.print_fn('[INIT] Starting system...')
        

            
        self.labeled_path = f'sessions/{session_name}/data/labeled_data.p'
        self.unlabeled_path = f'sessions/{session_name}/data/unlabeled_data.p'
        self.models_path = f'sessions/{session_name}/models/models.joblib'
        self.highlighter_path = f'sessions/{session_name}/models/highlighter.joblib'
        self.iteration_counter_path = f'sessions/{session_name}/data/iteration_counter.p'


        
        if os.path.isfile(self.iteration_counter_path):
            self.iteration_count = pickle.load(open(self.iteration_counter_path,'rb'))
            logging.debug(f'Iteration counter retrieved from disk, iteration count={self.iteration_count}')
        else:
            self.iteration_count = 0
                                     
        logging.debug(f'System starting. Session name={session_name}. Iteration count={self.iteration_count}')
        
        self.remaining_relevant=None
        self.estimated=False
        self.rangen = np.random.default_rng(2022)
        self.suggestions=[]
        self.annotations = pd.DataFrame([],columns=['label'])
#         self.labeling_batch=labeling_batch
                                     
        self.print_fn('[INIT] Loading data...')
        self._load_data(from_scratch)
        
        # REMOVE!!!
#         logging.warning('CÓDIGO DE DEBUG QUE FALTA ELIMINAR')
#         self.rangen.shuffle(self.unlabeled_data)
#         self.unlabeled_data = self.unlabeled_data[:50000]
        
        if not from_scratch:
            logging.debug(f"len(labeled_data)={len(self.labeled_data)} {self._labeled_data_str()}")
            logging.debug(f"len(unlabeled_data)={len(self.unlabeled_data)} {self._unlabeled_data_str()}")

            ################################
            # PRELOAD LABELED DATA VECTORS #
            ################################
            for item in self.labeled_data:
                item.preload_vector(type_=DataItem.TYPE_GLOVE300) 
                item.preload_vector(type_=DataItem.TYPE_BOW)       


            self.print_fn('[INIT] Loading/training models...')
            self._load_models()
            self._launch_estimator()
            self.print_fn('[INIT] System started.')
    #         self.save()
    #         self.status()                                     
#             logging.debug(' # # #. # # # ')
            logging.debug('#'*30+'   END INIT  '+'#'*30)
#             self.print_fn('[INIT] System started. Saving...[OK]')
            if not self.finish_function is None:
                self.finish_function()
#             logging.debug('#'*30+'END INIT'+'#'*30)

#     def set_confidence_value(self, value):
#         self.confidence_value = value


    def _launch_estimator(self):
        def thread_function():
            self.lock.acquire()
            self.thread_count+=1
            self.lock.release()
            
            self.ui.update_color_estimated_recall('red')
#             estimated = self.estimated_recall()*100
            lower, estimated, upper = self.estimated_recall()
            rel_count = self.get_relevant_count()
            lower_rel_est_count = (rel_count-lower*rel_count)/lower
            str_ = f'{lower*100:4.2f} % ~ {upper*100:4.2f} % (~{int(lower_rel_est_count):,})'
            self.ui.update_recall_estimator(str_)
            
            self.lock.acquire()
            self.thread_count-=1
            self.lock.release()
            if self.thread_count==0:
                self.ui.update_color_estimated_recall('green')  
            

        # Thread launch linked to widget 
        x = threading.Thread(target=thread_function)
        x.start()

    def _optimized_logreg_prediction(self, item_list, batch_size=80530):
        linear_model = self.term_highlighter.model
        vecnames = [item._vector_filename() for item in item_list]

        yhat = np.zeros(shape=(len(vecnames)))
        yhats = []
        for ini in range(0,len(vecnames), batch_size):
            fin = min(ini+batch_size, len(vecnames))
            X = normalize(sparse.vstack(map(lambda filename: pickle.load(open(filename, 'rb'))[3], vecnames[ini:fin])),axis=1)
            yhats.append(linear_model.predict_proba(X)[:,1])
        return np.hstack(yhats)
    def _estimated_relevant_u(self):
        return self._optimized_logreg_prediction(self.unlabeled_data)
#         unlabeled_vecnames = [item._vector_filename() for item in self.unlabeled_data]
#         linear_model = self.term_highlighter.model
#         yhat = np.zeros(shape=(len(unlabeled_vecnames)))
#         yhats = []
#         batch_size=80530
#         for ini in range(0,len(unlabeled_vecnames), batch_size):
#             fin = min(ini+batch_size, len(unlabeled_vecnames))
#             X = normalize(sparse.vstack(map(lambda filename: pickle.load(open(filename, 'rb'))[3], unlabeled_vecnames[ini:fin])),axis=1)
#             yhats.append(linear_model.predict_proba(X)[:,1])
#         return np.hstack(yhats)
#     def _estimated_relevant_u(self):
#         return self._optimized_logreg_prediction(self.unlabeled_data)
#         unlabeled_vecnames = [item._vector_filename() for item in self.unlabeled_data]
#         linear_model = self.term_highlighter.model
#         yhat = np.zeros(shape=(len(unlabeled_vecnames)))
#         yhats = []
#         batch_size=80530
#         for ini in range(0,len(unlabeled_vecnames), batch_size):
#             fin = min(ini+batch_size, len(unlabeled_vecnames))
#             X = normalize(sparse.vstack(map(lambda filename: pickle.load(open(filename, 'rb'))[3], unlabeled_vecnames[ini:fin])),axis=1)
#             yhats.append(linear_model.predict_proba(X)[:,1])
#         self.full_yhat=np.hstack(yhats)
#         return self.full_yhat
    
    def _estimated_relevant_r(self):
        
        return self._optimized_logreg_prediction(self.labeled_data)
#         labeled_vecnames = [item._vector_filename() for item in self.labeled_data]
#         linear_model = self.term_highlighter.model
#         yhat = np.zeros(shape=(len(labeled_vecnames)))
#         yhats = []
#         batch_size=80530
#         for ini in range(0,len(labeled_vecnames), batch_size):
#             fin = min(ini+batch_size, len(labeled_vecnames))
#             X = normalize(sparse.vstack(map(lambda filename: pickle.load(open(filename, 'rb'))[3], labeled_vecnames[ini:fin])),axis=1)
#             yhats.append(linear_model.predict_proba(X)[:,1])
#         return np.hstack(yhats)
#     def estimated_recall(self):
#         Rr = self._estimated_relevant_r()
#         Ur = self._estimated_relevant_u()
#         return Quant_estimator(Rr, Ur)
    def estimated_recall(self):
        Rr = self._estimated_relevant_r()
        Ur = self._estimated_relevant_u()
        lower, upper = QuantCI_estimator(Rr, Ur)
        return lower, Quant_estimator(Rr, Ur), upper
    
    def get_status(self):
        return self.status
    def get_labeled_count(self):
        return len(self.labeled_data)
    def get_unlabeled_count(self):
        return len(self.unlabeled_data)
    def get_relevant_count(self):
        return len([item for item in self.labeled_data if item.is_relevant()])
    def get_irrelevant_count(self):
        return len([item for item in self.labeled_data if  item.is_irrelevant()])
        
    #######################
    # LOAD MODELS ON INIT #
    #######################
    def _load_models(self):
        if not os.path.isfile(self.models_path):
            logging.debug('Creating the model from scratch. Need training.')
            assert not os.path.isfile(self.highlighter_path)
#             self.classifiers = [Classifier(SVC(kernel='linear',
#                                                C=1,
#                                                class_weight='balanced',
#                                                probability=True,
#                                                random_state=np.random.RandomState(42)), type_=DataItem.TYPE_BOW),
#                                 Classifier(SVC(kernel='rbf', 
#                                                C=15, 
#                                                class_weight='balanced',
#                                                probability=True,
#                                                random_state=np.random.RandomState(42)), type_=DataItem.TYPE_GLOVE300),]
            self.classifiers = [Classifier(MLPClassifier(early_stopping=True,
                                                         n_iter_no_change=20,
                                                         max_iter=1500,
                                                         hidden_layer_sizes=(20,), 
                                                         solver='adam', 
                                                         random_state=np.random.RandomState(42)), type_=DataItem.TYPE_BOW),
                                Classifier(MLPClassifier(early_stopping=True,
                                                         n_iter_no_change=20,
                                                         max_iter=1500,
                                                         hidden_layer_sizes=(100,), 
                                                         solver='adam', 
                                                         random_state=np.random.RandomState(42)), type_=DataItem.TYPE_GLOVE300),
                          ]

            self.term_highlighter = TermHighlighter()

            logging.debug('Starting training of models.')
            self._retrain(partial=False)
            logging.debug('Finished training of models.')
            
        else:
            logging.debug('Loading models from disk. No need for training.')
            assert os.path.isfile(self.highlighter_path)
            self._models_fromdisk()
            ## LEGACY BEHAVIOUR TO HAVE COMPATBILITY BACKWARDS.
            if self.term_highlighter.trained and not hasattr(self.term_highlighter, 'mean_value_feature'): 
                # trained but without info to compute new highlighting
                self.term_highlighter.compute_mean_value_feature(item_list=self.labeled_data,X=None)
            
    def _load_data(self,from_scratch):
        ################
        # LABELED DATA #
        ################
        if not from_scratch:
            if not os.path.isfile(self.labeled_path):
                logging.debug('Computing labeled data from "labeled_data.csv"')

                self.labeled_data=[]
                for line in open('labeled_data.csv').read().splitlines()[1:]:
                    id_,label = line.split(';')
                    item = DataItem(id_)
                    if label=='R':
                        item.set_relevant()
                    else:
                        item.set_irrelevant()
                        assert label=='I'
                    if item.has_vector():
                        self.labeled_data.append(item)
            else:
                logging.debug(f'Retrieving labeled data from disk ({self.labeled_path})')
                self.labeled_data = pickle.load(open(self.labeled_path,'rb'))
        else:
            self.labeled_data=[]
        ##################
        # UNLABELED DATA #
        ##################
        GM1 = '/home/ec2-user/SageMaker/data/GM_all_1945_1956/'
        GM2 = '/home/ec2-user/SageMaker/data/GM_all_1957-1967/'
        if not os.path.isfile(self.unlabeled_path):
            logging.debug(f'Computing unlabeled data from {GM1} and {GM2}')
            self.unlabeled_data = [DataItem(GM1+file_) for file_ in os.listdir(GM1)] + [DataItem(GM2+file_) for file_ in os.listdir(GM2)]
            relevant_ids = set([item.id_ for item in self.labeled_data])
            self.unlabeled_data = [item for item in self.unlabeled_data if item.has_vector() and not item.id_ in relevant_ids]
            
        else:
            #############################################
            # RETRIEVING FROM DISK INSTEAD OF COMPUTING #
            #############################################
            logging.debug(f'Retrieving unlabeled data from disk ({self.unlabeled_path})')
            self.unlabeled_data = pickle.load(open(self.unlabeled_path,'rb'))
            
        
        self.rangen.shuffle(self.unlabeled_data)
        
        if from_scratch: 
            logging.debug('Creating labeled data from scratch')
            self._from_scratch()
#             self.labeled_data=[]
#             valid=False
#             while not valid:
#                 seed = input('Please insert URLs to relevant document (; sep) (e.g., https://www.proquest.com/docview/1288605023/...)')
                
#                 matches = re.findall('docview/([0-9]*)/',seed)
#                 logging.debug(f'User inputs: {seed}')
#                 if len(matches)>=1:
#                     ids = set(matches)
#                     positions = [idx for idx,item in enumerate(self.unlabeled_data) if item.id_ in ids]
#                     if len(positions)>=1:
#                         for position in reversed(positions):
#                             self.labeled_data.append(self.unlabeled_data[position])
#                             self.labeled_data[-1].set_relevant()
#                             del(self.unlabeled_data[position])
#                         valid=True
#                         loging.debug(f"Valid input, documents found: {','.join([item.id_ for item in self.labeled_data])}")
#                     else:
#                         logging.debug('User query had something that look like valid documents but not present in database.')
#                         warning('Documents not found in database (The Globe and Mail 1936 onwards), please try again.')
#                 else:
#                     logging.debug('User input does not look like valid URL.')
#                     warning('Invalid URLs, please try again.')

        if self.debug:
            max_=1000
            self.unlabeled_data = self.unlabeled_data[:min(max_,len(self.labeled_data))]
#             self.labeled_data = self.labeled_data[:min(max_,len(self.labeled_data))]

        logging.debug(f'len(unlabeled_data)={len(self.unlabeled_data)}')
        logging.debug(f'len(labeled_data)={len(self.labeled_data)}')
    def _from_scratch(self):
        def iniatialize_from_input(input_):
            seed = input_

            matches = re.findall('docview/([0-9]*)/',seed)
            logging.debug(f'User inputs: {seed}')
            if len(matches)>=1:
                ids = set(matches)
                positions = [idx for idx,item in enumerate(self.unlabeled_data) if item.id_ in ids]
                if len(positions)>=2:
                    self.labeled_data=[]
                    for position in reversed(positions):
                        self.labeled_data.append(self.unlabeled_data[position])
                        self.labeled_data[-1].set_relevant()
                        del(self.unlabeled_data[position])
                        
                    valid=True
                    logging.debug(f"Valid input, documents found: {','.join([item.id_ for item in self.labeled_data])}")
                    logging.debug(f"len(labeled_data)={len(self.labeled_data)} {self._labeled_data_str()} (FROM_SCRATCH)")
                    logging.debug(f"len(unlabeled_data)={len(self.unlabeled_data)} {self._unlabeled_data_str()}")

                    ################################
                    # PRELOAD LABELED DATA VECTORS #
                    ################################
                    for item in self.labeled_data:
                        item.preload_vector(type_=DataItem.TYPE_GLOVE300) 
                        item.preload_vector(type_=DataItem.TYPE_BOW)       


                    self.print_fn('[INIT] Loading/training models...')
                    self._load_models()
                    self.print_fn('[INIT] System started. Saving...')
            #         self.save()
            #         self.status()                                     
                    logging.debug(' # # # System started succesfully. # # # ')

                    self.print_fn('[INIT] System started. Saving...[OK]')
                    if not self.finish_function is None:
                        self.finish_function()
                    logging.debug('#'*30+'END INIT'+'#'*30)
                else:   
                    logging.debug('User query had something that look like valid documents but not present in database '\
                                  f'({len(positions)} found).')
                    if len(positions)==0:
                        warning('Documents not found in database (The Globe and Mail 1936 onwards), please try again.')
                    else:
                        warning('At least two documents required (in The Globe and Mail 1936 onwards), please try again.')
                    self._from_scratch()
            else:
                logging.debug('User input does not look like valid URL.')
                warning('Invalid URLs, please try again.')
                self._from_scratch()
        self.print_fn('[INIT] Please insert URLs to relevant document (; sep) (e.g., https://www.proquest.com/docview/1314032787/'\
              ';https://www.proquest.com/docview/1289363937/...)')
        request_user_input(iniatialize_from_input)
                                     

    ################################################################ 
    #                        SAVE LISTS                            #   
    ################################################################        
    def save(self):   
        """
        Saves the current state of the system, including labeled data, unlabeled data, 
        and iteration count, to disk. This method performs the following steps:
        
        1. Asserts that there are no pending suggestions to be moved to labeled data.
        2. Logs debug messages regarding the saving process and the creation or overwriting
           of relevant files.
        3. Serializes and saves the labeled data, unlabeled data, and iteration count to 
           their respective file paths using `pickle`.
        4. Invokes the `_models_todisk` method to save the models to disk.
        5. Displays a message indicating the save process has completed.


        Raises:
            AssertionError: If there are pending suggestions in `self.suggestions`, as this 
                            method expects no suggestions to be pending before saving.
        """
        assert len(self.suggestions)==0
#         if len(self.suggestions)>0:
#             logging.debug('Attemping to save system\'s state but there are labeled suggestions to be stored first. Re-train required.')
#             self._move_suggestions_to_labeled()
#         else:
        logging.debug('#'*30+'  SAVE  '+'#'*30)
        logging.debug('Saving system\'s state. NO suggestions pending to be moved to labeled data.')
                                     
        for file_ in [self.labeled_path,self.unlabeled_path,self.iteration_counter_path]:
            if os.path.isfile(file_):
                logging.debug(f'Overwriting file {file_}')
            else:
                logging.debug(f'Creating file {file_}')
        self.print_fn('[SAVE] Saving system\'s state...')                
        pickle.dump(self.labeled_data, open(self.labeled_path, 'wb'))
        pickle.dump(self.unlabeled_data, open(self.unlabeled_path, 'wb'))
        pickle.dump(self.iteration_count, open(self.iteration_counter_path,'wb'))
        self._models_todisk()           
        self.print_fn('[SAVE] Saving system\'s state...[DONE]')     
#         logging.debug('#'*30+'END SAVE'+'#'*30)              
    ################################################################
    #                   save/load models                           #
    ################################################################  
    def _models_todisk(self):
        for file_ in [self.models_path,self.highlighter_path ]:
            if os.path.isfile(file_):
                logging.debug(f'Overwriting file {file_}')
            else:
                logging.debug(f'Creating file {file_}')
        dump(self.classifiers, self.models_path)
        dump(self.term_highlighter, self.highlighter_path)
                                     
    def _models_fromdisk(self):
        logging.debug(f'Retrieving {self.highlighter_path} and {self.models_path} from disk')
        self.term_highlighter = load(self.highlighter_path)
        self.classifiers = load(self.models_path)
           

    def _search_suggestions(self, all_=False, progress_bar=False, one_iteration=False):
            # MAKE PREDICTIONS AND STORE SUGGESTIONS....
#             print('Searching over the all dataset')
            
            batch = self.suggestion_sample_size
            logging.debug(f'Search over the all dataset (all_={all_}) using batchs of size {batch} one_iteration={one_iteration}.')
#             all_batches = [(i,min(i+batch,len(self.unlabeled_data)))  for i in range(0, len(self.unlabeled_data),batch)]
#             all_batches = [f'[{ini}, {fin})' for ini,fin in all_batches]
#             logging.debug(f'Analyzing the following batches: '+' '.join(all_batches))
#             suggestions = []
            candidate_args=[]
            yhat=[]
            if progress_bar: 
                progress_bar_widget = widgets.IntProgress(value=0,
                                                          min=0,
                                                          max=len(self.unlabeled_data),
                                                          description='Exporting:',
                                                          bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
                                                          style={'bar_color': 'green'},
                                                          orientation='horizontal'
                                                          )
                display(progress_bar_widget)
            for i in range(0, len(self.unlabeled_data),batch):
                ini = i
                fin = min(ini+batch,len(self.unlabeled_data))
#                 print(f'[{ini},{fin})') 
                logging.debug(f'Analyzing batch '+f'[{ini}, {fin})')
                args = list(range(ini,fin))
                candidates,predictions = self._filter_and_sort_candidates(args)
                candidate_args += list(candidates)
                logging.debug(f'Number of suitable candidates found: {len(candidate_args)}')
                yhat += predictions
#                 candidate_args, yhat += self._filter_and_sort_candidates(candidate_args)
#                 suggestions += list(zip([self.unlabeled_data[arg] for arg in candidate_args], yhat))
                if not all_ and len(candidate_args)>0:
                    logging.debug(f'Suitable candidates found and all_={all_}, so, stopping search.')
                    break
#                     print('We found something, stopping...')
                if progress_bar:
                    progress_bar_widget.value=fin
                if one_iteration:
                    logging.debug(f'Stopping after one iteration (one_iteration={one_iteration}).')
                    break
            cap=fin
            if cap!=(len(self.unlabeled_data)):
                self.estimated=True
            self.remaining_relevant = len(candidate_args) 
            logging.debug(f'Potentially relevant found in the {cap} articles analyzed (during search): {self.remaining_relevant} ')
            if self.estimated:
                self.remaining_relevant = int((self.remaining_relevant/cap)*len(self.unlabeled_data))
                logging.debug(f'Estimated relevant in the remaining {len(self.unlabeled_data)} articles: {self.remaining_relevant} ')
            else:
                logging.debug(f'The search was conducted over the all dataset, the number of candidates found is final, not estimated')
            return candidate_args,yhat
#             suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)           
    ################################################################ 
    #                             LOOP                             #   
    ################################################################  
    def loop(self, suggestion_sample_size=1000, labeling_batch_size=10, confidence_value=0.5, full_search=True, one_iteration=False ): 
        """
        Runs an active learning loop that generates suggestions, annotates them, and retrains the model 
        with the newly labeled data.

        The method performs the following steps:
        1. Verifies the loop state and prepares data for processing.
        2. Configures loop parameters like batch size, confidence value, and suggestion sample size.
        3. Computes new suggestions for labeling based on the current model state.
        4. Presents new suggestions for user annotation via the `pixt.annotate` method.
        5. Moves labeled suggestions to the labeled data pool and triggers retraining of the model.
        6. Calls a `finish_function` after the annotation process is completed.

        Args:
            suggestion_sample_size (int): Number of suggestions to sample per iteration. Default is 1000.
            labeling_batch_size (int): Number of samples to label per batch. Default is 10.
            confidence_value (float): Minimum confidence score for a suggestion to be considered. Default is 0.5.
            full_search (bool): Whether to perform a full search for new suggestions or a limited one. Default is True.
            one_iteration (bool): If True, only a single loop iteration will be executed. Default is False.

        Raises:
            AssertionError: If there are issues with the loop state or if duplicate IDs are found in the data.

        Notes:
            The method relies on a `finish_function` and a `cancel_loop` function being defined for post-processing.
        """
        self.print_fn('[LOOP] Starting loop...')
        assert len(self.suggestions)==0 and not self.loop_started # if a loop was interrupted in the middle, the suggestions must be moved
                                                                  # to the unlabeled_data list
        ids_list = [item.id_ for item in self.labeled_data]
        ids_list += [item.id_ for item in self.unlabeled_data]
        ids_list += [item.id_ for item in self.suggestions]

        assert not has_duplicated(ids_list)
        del(ids_list)
                                  
        self.labeling_batch=labeling_batch_size
        self.loop_started=True
        self.confidence_value = confidence_value
        self.suggestion_sample_size = suggestion_sample_size
        self.labeling_batch_size=labeling_batch_size
                                  
        logging.debug('#'*30+f'  LOOP {self.iteration_count:3}  '+'#'*30)
#         logging.debug('#'*30+f'END LOOP {self.iteration_count:3}'+'#'*30)       
        ####################################################
        # MOVING FROM SUGGESTIONS (ANNOTATIONS) TO LABELED #
        ####################################################
        def after_labeling():
            if len(self.suggestions)>0:
                print('Starting retraining...')
                self._move_suggestions_to_labeled()
            if not self.finish_function is None:
                self.finish_function()
            if self.loop_started: # LOOP NOT CANCELLED
                logging.debug('#'*30+f'END LOOP {self.iteration_count:3}'+'#'*30) 
                self.iteration_count+=1
                self.loop_started=False
        self.print_fn('[LOOP] Calculating new suggestions...')
        self._compute_suggestions(full_search=full_search, one_iteration=one_iteration)
        self.print_fn('[LOOP] ')

        if len(self.suggestions)>0:
            highlighter = None
            if self.term_highlighter.trained:
                logging.debug(f"Highlighting with {','.join(self.term_highlighter.sorted_terms()[:10])},...")
                highlighter = self.term_highlighter
            text_for_label = [suggestion.get_htmldocview(highlighter=highlighter,confidence_score=confidence)
                              for suggestion,confidence in zip(self.suggestions,self.yhat)]

            print()
#             self.status()
            self.annotations = pixt.annotate(
                                             text_for_label,
                                             options=[HRSystem.RELEVANT_LABEL, HRSystem.IRRELEVANT_LABEL],
                                             stop_at_last_example=False,
                                             display_fn=html,
                                             cancel_process_fn=self.cancel_loop,
                                             final_process_fn=after_labeling
                                            )
#             logging.debug('#'*30+f' END LOOP {self.iteration_count:3}'+'#'*30)  
            
        else:
            self.loop_started=False
            logging.info('No new suggestions found, nothing to do on loop, skipping.')
            self.print_fn('[LOOP] No new suggestions found, nothing to do on loop, skipping.')
            self.finish_function()
                          
        
             
    def cancel_loop(self):
#         self.iteration_count-=1
        logging.debug('#'*30+f' LOOP CANCELLED (LOOP no. {self.iteration_count})'+'#'*30)
#         logging.debug(f'Reducing loop no ({self.iteration_count+1} - 1)={self.iteration_count}')
#         self.iteration_count-=1
        logging.debug(f'Clearing list of candidates len(candidates)={len(self.candidate_args)} ({self._candidates_str()}).')
        self.print_fn('[CNCL] Canceling LOOP...')
        self.candidate_args = []
#         self.print_fn('[CNCL] No new suggestions found, nothing to do on loop, skipping.')
#         self.iteration_count-=1
                         


        self.unlabeled_data = self.unlabeled_data + self.suggestions
        self.suggestions=[]
        self.loop_started=False
                              
        logging.debug('Moving the suggestions made by the model from suggestions --to--> unlabeled_data')              
        logging.debug(f"new len(unlabeled_data)={len(self.unlabeled_data)} {self._unlabeled_data_str()}")
        logging.debug(f'new len(suggestions)={len(self.suggestions)} {self._suggestions_str()}')
                              
        self.print_fn('[CNCL] Canceling LOOP...[DONE]')
        logging.debug('#'*30+' END LOOP CANCELLED '+'#'*30)
                              
    def _filter_and_sort_candidates(self,candidate_args):                                     
        # First Model        
#         yhat1 = self.classifiers[0].predict([self.unlabeled_data[arg] for arg in candidate_args])
#         yhat1 = self.term_highlighter.predict([self.unlabeled_data[arg] for arg in candidate_args])                
        yhat1 = self._optimized_logreg_prediction([self.unlabeled_data[arg] for arg in candidate_args]) #self.term_highlighter.predict([self.unlabeled_data[arg] for arg in candidate_args])                
        candidate_args = np.array(candidate_args)[yhat1>self.confidence_value]
        logging.debug(f'Number of suggestions 1st model (highlighter): {np.sum(yhat1>self.confidence_value)} (discarded: {np.sum(yhat1<=self.confidence_value)})')
                         
        # Second Model
        yhat1 = yhat1[yhat1>self.confidence_value]
        candidate_args = np.array(candidate_args)[yhat1>self.confidence_value]
        if len(candidate_args)==0:
            return [],[]
        yhat1 = yhat1[yhat1>self.confidence_value]
        yhat2 = self.classifiers[1].predict([self.unlabeled_data[arg] for arg in candidate_args])   
        logging.debug(f'Number of suggestions 2nd model (GloVe NN): {np.sum(yhat2>self.confidence_value)} (discarded: {np.sum(yhat2<=self.confidence_value)})')
                                     
        # Third Model
        mask = (yhat2>self.confidence_value)
        yhat1 = yhat1[mask]
        yhat2 = yhat2[mask]
        candidate_args = np.array(candidate_args)[mask]
        if len(candidate_args)==0:
            return [],[]
#         yhat4 = self.term_highlighter.predict([self.unlabeled_data[arg] for arg in candidate_args])
        yhat4 = self.classifiers[0].predict([self.unlabeled_data[arg] for arg in candidate_args])
        logging.debug(f'Number of suggestions 3rd model (BOW NN): {np.sum(yhat4>self.confidence_value)} (discarded: {np.sum(yhat4<=self.confidence_value)})')
                                     
        # Average
        yhat = np.average(np.vstack([yhat1,yhat2,yhat4]), axis=0)                                     
        yhat = yhat[yhat4>self.confidence_value]
        candidate_args = np.array(candidate_args)[yhat4>self.confidence_value]
        candidate_args = np.array(candidate_args)[np.argsort(yhat)[::-1]]
        yhat = yhat[np.argsort(yhat)][::-1]
        if len(candidate_args)==0:
            return [],[]
               
        # Info for debugging
        #end = min(len(candidate_args), self.labeling_batch_size)
        confidence_levels = [f'{yhat[arg]:4.3f}' for arg in np.argsort(yhat)[::-1][:10]]
        logging.debug(f"Confidence levels for suggestions: {','.join(confidence_levels)} ")
                      
        return candidate_args, list(yhat)
  
    ################################################################
    #                          Re-Train                            #
    ################################################################  
    def _retrain(self, partial=True): 
        expanded=False
        if len(self.labeled_data)<10:

            expanded=True  
            expansion = len(self.labeled_data)*10
            self.labeled_data += [DataItem(item.id_) for item in self.rangen.choice(
                                                                                   self.unlabeled_data,
                                                                                   size=expansion,
                                                                                   replace=False)]
                                 
            logging.debug(f'Only {len(self.labeled_data)} articles labeled. Expanding labeled data'\
                         f' (adding {expansion} randomly selected negative examples)'\
                         f" ({','.join([str(item.id_) for item in self.labeled_data[-expansion:]])})")
                                     
            for item in self.labeled_data[-expansion:]:
                item.set_irrelevant()
        logging.debug(f'Training each of the {len(self.classifiers)} classifiers.')
        for clf in self.classifiers:
#             info(f'Training: {str(clf)}')
            logging.debug(f'Training model {str(clf)}')
            if not partial:
                clf.fit(self.labeled_data)
            else:
                clf.fit(self.labeled_data)
            
        logging.debug('Training term highlighter.')
        self.term_highlighter.fit(self.labeled_data)
        logging.debug('Done training.')
        self._launch_estimator()             
        
        if expanded:
                logging.debug('Removing randomly selected negatives examples (expansion)'\
                             f" ({','.join([str(item.id_) for item in self.labeled_data[-expansion:]])})")                                     
                
                self.labeled_data = self.labeled_data[:-expansion]
                logging.debug(f'Size of labeled data after removing expansion={len(self.labeled_data)}')
    
    def _extend_suggestions(self):
        ## ADD 10k candidates extracted from global recall estimations, ONLY if there are not candidates already and only if estimations
        ## available 
        ## REMOVE THE 10k candidates from global recall estimations
        pass
    def _compute_suggestions(self, full_search=True, one_iteration=False):
        cap = min(self.suggestion_sample_size, len(self.unlabeled_data))
                                     
        self.estimated=False
        if cap!=len(self.unlabeled_data):
                  self.estimated=True
                                     
        logging.debug(f'Computing batch of suggestions, batch_size={cap}')
        if len(self.candidate_args)!=0:
            # WORKING WITH PREVIOUS BATCH OF CANDIDATES
            logging.debug(f'There are candidate {len(self.candidate_args)} args remaining to be analyzed. Computing model predictions...')
#             self._extend_suggestions()
            self.candidate_args, self.yhat = self._filter_and_sort_candidates(self.candidate_args)
            logging.debug(f'Suitable candidates after filtering {len(self.candidate_args)}.')
        if len(self.candidate_args)==0:
            logging.debug(f'There are no suitable candidates to be analyzed.')
            logging.debug(f'Using new random batch size {cap} and computing model predictions...')
            # IF NO NEW SUGGESTIONS IN PREVIOUS BATCH ( OR PEVIOUS BATCH EMPTY ALL TOGETHER), CREATING NEW BATCH
            self.candidate_args = self.rangen.choice(range(len(self.unlabeled_data)), size=cap, replace=False )
            self._extend_suggestions()
            self.candidate_args, self.yhat = self._filter_and_sort_candidates(self.candidate_args)
            logging.debug(f'Suitable candidates for new batch after filtering {len(self.candidate_args)}.')
#         else:
#             # FILTERING OLD BATCH
#             self.candidate_args, _ = self._filter_and_sort_candidates(self.candidate_args)
#             if len(self.candidate_args)==0: 
#                 # IF NO GOOD SUGGESTION THEN CREATING NEW BATCH
#                 self.candidate_args = self.rangen.choice(range(len(self.unlabeled_data)), size=cap, replace=False )  
#                 self.candidate_args, _ = self._filter_and_sort_candidates(self.candidate_args)
                      
        if len(self.candidate_args)==0 and full_search:
            logging.debug('No suitable candidates found randomly (not using previous nor new random batch). Searching over all corups...')
#             print('No good suggestions found randomly.')
            self.candidate_args, self.yhat = self._search_suggestions(all_=False, one_iteration=one_iteration)
            logging.debug(f'{len(self.candidate_args)} suitable candidates found.')
            logging.debug(f'Unable to compute number of remaining relevants (estimated).')
        else: 
            self.remaining_relevant = len(self.candidate_args) 
            logging.debug(f'Relevant found in the {cap} articles analyzed: {self.remaining_relevant} ')
            if self.estimated:
                self.remaining_relevant = int((self.remaining_relevant/cap)*len(self.unlabeled_data))
                logging.debug(f'Estimated relevant in the remaining {len(self.unlabeled_data)} articles: {self.remaining_relevant} ')

        
        ################
        ## UNTIL HERE ##
        ################
        
        self.suggestions = []
        self.annotations = []
        
        if self.remaining_relevant==0:
            logging.debug(f'No good candidates (predictions<{self.confidence_value}).')
            self.print_fn('[LOOP] There are no good candidates provided by the model. '\
            'This could happend at the beginin and at the end of the labeling process')
        else:    
            end = min(len(self.candidate_args),self.labeling_batch_size)
#             self.candidate_args = self.candidate_args[np.argsort(self.yhat)][::-1]
#             self.yhat = np.array(self.yhat)[np.argsort(self.yhat)][::-1]
            best_ten_args = self.candidate_args[:end]

            for arg in best_ten_args:
                self.suggestions.append(self.unlabeled_data[arg])

            for arg in sorted(best_ten_args,reverse=True):
                del(self.unlabeled_data[arg])
                self.candidate_args = [arg2 if arg2<arg else arg2-1 for arg2 in self.candidate_args if arg2!=arg]
                # REMOVE ALSO FROM YHAT!!!!
                      

            logging.debug('Moving the suggestions made by the model from unlabeled_data --to--> suggestions')              
            logging.debug(f"new len(unlabeled_data)={len(self.unlabeled_data)} {self._unlabeled_data_str()}")
            logging.debug(f'new len(suggestions)={len(self.suggestions)} {self._suggestions_str()}')

            for classifier in self.classifiers:
                for item in self.suggestions:
                    item.preload_vector(type_=classifier.vector_type)
                    item.preload_vector(type_=classifier.vector_type)
                      
    ################################################################ 
    #                            EXPORT                            #   
    ################################################################  
    def export(self, suggestion_sample_size=1000,  confidence_value=0.5, send_email=False, compute_suggestions=False):
        """
        Exports relevant articles from the labeled and potentially suggested data to a CSV file and optionally sends the file via email.

        This method performs the following steps:
        
        1. Logs the start of the export process and validates that no suggestions are pending.
        2. Writes labeled data that are marked as relevant to a CSV file.
        3. Optionally computes suggestions for unlabeled data using the model and writes those to the CSV file as well.
        4. Optionally sends the CSV file to a specified S3 bucket via email using AWS S3, after creating a temporary copy.
        
        Parameters:
            suggestion_sample_size (int): The number of suggestions to compute (default is 1000). Only used if `compute_suggestions` is True.
            confidence_value (float): The confidence threshold for considering suggestions (default is 0.5). Only used if `compute_suggestions` is True.
            send_email (bool): Whether to send the exported file via email (default is False).
            compute_suggestions (bool): Whether to compute suggestions based on unlabeled data (default is False).
        
        Raises:
            AssertionError: If there are pending suggestions in `self.suggestions` (this method expects no suggestions to be pending before exporting).
            FileNotFoundError: If the generated file cannot be found when sending the email.
        
        Notes:
            - The CSV file will contain columns for `URL`, `relevant_or_suggested` (either 'rel' or 'sugg'), and `confidence`.
            - The file is saved to the `sessions/{self.session_name}/data/` directory with a timestamped filename.
            - If `compute_suggestions` is True, the model will generate suggestions for unlabeled data, which will be saved to the CSV with their respective confidence values.
            - If `send_email` is True, the file will be uploaded to an S3 bucket after being copied and sent.
        """
        logging.debug('#'*30+' EXPORT '+'#'*30)
        self.confidence_value = confidence_value
        self.suggestion_sample_size =  suggestion_sample_size
        assert len(self.suggestions)==0
        logging.debug('#'*30+'  EXPORT  '+'#'*30)
        self.print_fn('[LOOP] Starting export...')
#         if len(self.suggestions)>0:
#             logging.debug('Attemping to export system\'s state but there are labeled suggestions to be stored first. Re-train required.')
#             self._move_suggestions_to_labeled()
#         else:
        logging.debug('Exporting... (NO suggestions pending to be moved to labeled data)')
        count=0   
        filename = f'sessions/{self.session_name}/data/exported_data_'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'
        with open(filename, 'w') as writer:
            writer.write('URL,relevant_or_suggested,confidence\n')
            for item in self.labeled_data:
                if item.is_relevant():
                    count+=1
                    writer.write(f'https://proquest.com/docview/{item.id_},rel,1\n')
            

            
            logging.debug(f'Number of articles relevant founds {count} (from labeled)')
        
            # MAKE PREDICTIONS AND STORE SUGGESTIONS....
#             batch= 20000
#             suggestions = []
#             for i in range(0, len(self.unlabeled_data)+1,batch):
#                 ini = i
#                 fin = min(ini+batch,len(self.unlabeled_data))
#                 args = list(range(ini,fin))
#                 candidate_args, yhat = self._filter_and_sort_candidates(candidate_args)
            status_str = ''
            if compute_suggestions:   
                status_str = f'[LOOP] Number of articles relevant founds {count} (from labeled). Computing suggestions by the model...'
                self.print_fn(status_str)
                candidate_args, yhat = self._search_suggestions(all_=True, progress_bar=True)
                suggestions = list(zip([self.unlabeled_data[arg] for arg in candidate_args], yhat))
                suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
                count=0
                for item,confidence in suggestions:
                    count+=1
                    writer.write(f'https://proquest.com/docview/{item.id_},sugg,{confidence:4.3f}\n')
                status_str = f'[LOOP] Number of possible relevant article founds {count} (suggestions model) [OK]'
                self.print_fn(status_str)
                logging.debug(f'Number of possible relevant article founds {count} (suggestions model)')
            else:
                status_str = f'[LOOP] Number of articles relevant founds {count} (from labeled). Skipping suggestions [OK]'
                self.print_fn(status_str)
                      
        if send_email:
            assert os.path.isfile(filename)
            status_str+='. Sending email...'
            self.print_fn(status_str)
            temp_file ='exported_data_'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'
            shutil.copyfile(filename, temp_file)
            assert os.path.isfile(temp_file)
            os.system(f'aws s3 cp {temp_file} s3://pq-tdm-studio-results/tdm-ale-data/623/results/')
            os.remove(temp_file)
            logging.debug('E-mail sending enabled, email sent [OK].')
            status_str+='[OK]'
        else:
            status_str+='. Skipping sending email [OK]'
            logging.debug('E-mail sending disabled, skipping.')
        self.print_fn(status_str)              
        logging.debug('#'*30+' END EXPORT '+'#'*30)    
      
#         logging.debug('#'*30+'END EXPORT'+'#'*30)
    ################################################################ 
    #                            STATUS                            #   
    ################################################################ 
    def _center(str_,width=100):
        width_aux =width-2
        out =  '#'+' '*(int((width_aux-len(str_))/2))+str_+' '*(int((width_aux-len(str_))/2))+'#'
        if len(out)!=width:
            out = out[:-1]+' #'
        return out
    def _left(str_,width=100):
        output =  '# '+str_
        remaining = width-len(output)
        return output+' '*(remaining-1)+'#'
    def status(self):
        """
        Displays the status of the HRSystem, including details about labeled and unlabeled data, relevant and irrelevant articles,
        classifier performance metrics, and confusion matrix for each model in the system.

        The output includes:
        - System header with the current status title.
        - Summary of labeled and unlabeled articles, and their classification status (relevant or irrelevant).
        - Detailed performance metrics for each classifier and model, including training and testing accuracy, precision, recall, and F1-score.
        - The confusion matrix for each model showing True Negative (TN), False Positive (FP), False Negative (FN), and True Positive (TP) values.

        The status report also indicates if any suggestions are estimated and provides a summary of the classifiers' performance.
        Each model’s results are logged for further analysis.

        Logs:
            - Performance metrics (train and test) for each classifier.
            - Confusion matrix values for each model.

        """
        width = 100
        print('#'*width)

        print(HRSystem._center('~~~~~~~~~~~~~~~~~~'))
        print(HRSystem._center('~ System Status: ~'))
        print(HRSystem._center('~~~~~~~~~~~~~~~~~~'))
        print(HRSystem._left(f'Number of labeled articles:   {len(self.labeled_data):10,}    -    {self._relevant_count():10,} relevant '\
        f'   {len(self.labeled_data)-self._relevant_count():10,} irrelevants'))
        if self.remaining_relevant is None:
            print(HRSystem._left(f'Number of unlabeled articles: {len(self.unlabeled_data):10,}    -'+' '*10+' N/A suggestions '))
        else:
            print(HRSystem._left(f'Number of unlabeled articles:   {len(self.unlabeled_data):10,}  -    {self.remaining_relevant:10,} suggestions '\
            f'{len(self.unlabeled_data)-self.remaining_relevant:10,} irrelevants'))
            if self.estimated:
                print(HRSystem._left(' '*60+'(ESTIMATED)'))
        print(HRSystem._center(''))
        
        ytrue = DataItem.get_y(self.labeled_data)
        for model in self.classifiers + [self.term_highlighter]:
            print(HRSystem._left(str(model.model).replace('\n','').replace('  ','')))
            print(HRSystem._left('~'*len(str(model.model).replace('\n','').replace('  ',''))))

            yhat = model.predict(self.labeled_data)>0.5
            yhat = yhat.astype('int')
            scores = model.cross_validate_on(self.labeled_data,cv=3)
            metrics = [
                       scores['train_accuracy'],
                       scores['train_precision'],
                       scores['train_recall'],
                       scores['train_f1'],
                       ]
            performance = f'{str(model)};train_accuracy:{metrics[0]};train_precision:{metrics[1]};'
            performance += f'train_recall:{metrics[2]};train_f1:{metrics[3]};'
            logging.info(performance)
            print(HRSystem._left('accuracy   precision      recall      f1-score'))
            print(HRSystem._left('  '+'      '.join(f'{metric:5.4f}' for metric in metrics) +' (train)'))
            metrics = [
                       scores['test_accuracy'],
                       scores['test_precision'],
                       scores['test_recall'],
                       scores['test_f1'],
                       ]
            performance = f'{str(model)};test_accuracy:{metrics[0]};test_precision:{metrics[1]};'
            performance +=f'test_recall:{metrics[2]};test_f1:{metrics[3]};'
            logging.info(performance)
            print(HRSystem._left('  '+'      '.join(f'{metric:5.4f}' for metric in metrics)+' (test )'))
            tn, fp, fn, tp = (confusion_matrix(ytrue,yhat)).ravel()
            print(HRSystem._center('~~~~~~~~~~~~~~~~~~~~~~~~~~'))
            print(HRSystem._center('~~~~~Confusion Matrix~~~~~'))
            print(HRSystem._center('~~~~~~~~~~~~~~~~~~~~~~~~~~'))
            print(HRSystem._center(f'TN = {tn:6,}    FP = {fp:6,}'))
            print(HRSystem._center(f'FN = {fn:6,}    TP = {tp:6,}'))
            print(HRSystem._center(' '))
        print('#'*width)
    def _labeled_data_str(self):
        if len(self.labeled_data)>4:
            str_ = '<'+','.join([item.id_ for item in self.labeled_data[:2]])
            str_ +=', ... ,'
            str_ +=','.join([item.id_ for item in self.labeled_data[-2:]])+'>'
        else:
            str_ ='<'+','.join([item.id_ for item in self.labeled_data])+'>'
                      
        return str_
    def _unlabeled_data_str(self):
        if len(self.unlabeled_data)>4:
            str_ = '<'+','.join([item.id_ for item in self.unlabeled_data[:2]])
            str_ +=', ... ,'
            str_ +=','.join([item.id_ for item in self.unlabeled_data[-2:]])+'>'
        else:
            str_ ='<'+','.join([item.id_ for item in self.unlabeled_data])+'>'
        return str_
                      
    def _suggestions_str(self,full_list=False):
        if len(self.suggestions)<=4 or full_list:
            str_ = '<'+','.join([item.id_ for item in self.suggestions])+'>'
        else:
            str_ = '<'+','.join([item.id_ for item in self.suggestions[:2]])
            str_ +=', ... ,'
            str_ +=','.join([item.id_ for item in self.suggestions[-2:]])+'>'
            
        return str_
    def _candidates_str(self):
        if len(self.candidate_args)>4:
            str_ = '<'+','.join([str(arg) for arg in self.candidate_args[:2]])
            str_ +=', ... ,'
            str_ +=','.join([str(arg) for arg in self.candidate_args[-2:]])+'>'
        else:
            str_ = '<'+','.join([str(arg) for arg in self.candidate_args])+'>'
        return str_          
    def _relevant_count(self):
        return len([item for item in self.labeled_data if item.label==DataItem.REL_LABEL])    
                 
                      
                      

    def _move_suggestions_to_labeled(self, skip_retrain=False): 
        assert len(self.suggestions)>0

        logging.debug(f'There are {len(self.suggestions)} suggestions labeled that need to be moved to labeled_data. Re-train required.')
        logging.info(f'len(suggestions)={len(self.suggestions)} {self._suggestions_str(full_list=True)}')
        logging.info(f'Annotations: '+','.join(self.annotations["label"]))
        logging.debug('Moving from suggestions --to--> labeled data (for latter training)')
        
        relevant_count=0
            
        for item,label in zip(self.suggestions, self.annotations["label"]):
            if label==HRSystem.RELEVANT_LABEL:
                item.set_relevant()
                relevant_count+=1
            else:
                item.set_irrelevant()
                assert label==HRSystem.IRRELEVANT_LABEL
        logging.info(f'From the {len(self.suggestions)} suggestions {relevant_count} where found relevant.'\
                     f' ({relevant_count/len(self.suggestions):5.4f})')
        
        self.labeled_data = self.labeled_data+self.suggestions
        logging.debug(f"new len(labeled_data)={len(self.labeled_data)} {self._labeled_data_str()}")
        logging.debug(f"new len(unlabeled_data)={len(self.unlabeled_data)} {self._unlabeled_data_str()}")
                                     
        del(self.annotations)
        self.suggestions=[]
        if not skip_retrain:
            self.print_fn('[LOOP] Re-training models using new suggestions...')
            logging.debug(f'Re-training model ussing new {len(self.suggestions)} suggestions.')
            self._retrain()
            self.print_fn('[LOOP] Re-training models using new suggestions...[OK]')
    
                      
    def review_labeled(self, how_many=20):
        """
        Reviews a specified number of labeled items (default is 20), allowing for corrections to their labels. After reviewing, 
        it retrains the model if changes are made to the labels. The method displays the reviewed items, compares the original and 
        new labels, and logs the results.

        Parameters:
            how_many (int): The number of labeled items to review (default is 20). It limits the review to the most recent items in the labeled data.

        Raises:
            AssertionError: If there are pending suggestions in `self.suggestions`, as this method expects no suggestions to be pending before reviewing.

        Behavior:
            - Displays the labeled items to be reviewed, providing a mechanism for annotators to correct labels.
            - Compares the existing labels with the corrected labels and counts the number of changes.
            - If any changes are made, it logs the changes, retrains the model using the updated labels, and provides feedback to the user.
            - If no changes are made, it simply logs that no retraining is needed.
            - Calls a `finish_function`, if defined, after the review process is complete.

        Workflow:
            1. Asserts that there are no pending suggestions to be reviewed.
            2. Displays the most recent `how_many` labeled items for review.
            3. Annotates the items using an external tool (`pixt.annotate`).
            4. Upon completing the review, compares the labels and logs any changes.
            5. If changes are made, retrains the model using the updated labels.
            6. Optionally calls a `finish_function` after completing the review process.

        Notes:
            - The method assumes that the annotations are either relevant or irrelevant, and these are defined by the system labels.
            - The retraining step occurs only if there are changes to the labels; otherwise, no retraining is performed.
            - The reviewed data is passed to an external annotation tool (`pixt.annotate`) for manual or automated labeling.
        """
        self.print_fn(f'[REVIEW] Reviewing {how_many} items...')
        logging.debug(f'{how_many} items are being REVISED...')
        def after_reviewing():
            changes=0
            for item,label in zip(self.labeled_data[-how_many:], self.annotations["label"]):
                old_label=item.is_relevant()
                if label==HRSystem.RELEVANT_LABEL:
                    item.set_relevant()
                else:
                    item.set_irrelevant()
                if old_label!=item.is_relevant():
                      changes+=1
            logging.debug(f'Labels were reviewed ({how_many} items were analyzed).')
                      
            if changes>0:
                logging.debug(f'CHANGES WERE MADE during reviewing. There were {changes} changes. Logging results and retraining...')
                      
                # CORREGIR
                logging.info(f'len(revisions)={len(self.labeled_data[-how_many:])} '\
                                '<'+','.join([str(item.id_) for item in self.labeled_data[-how_many:]])+'>')
                # CORREGIR
                      
                logging.info(f'Annotations(revisions): '+','.join(self.annotations["label"]))
                self.print_fn(f'[REVIEW] Changes were made ({changes} changes). Re-training models using updated labels...')
                logging.debug(f'Re-training model updated labels {changes}.')
                self._retrain()
                self.print_fn(f'[REVIEW] Changes were made ({changes} changes). Re-training models using updated labels...[OK]')
            else:
                logging.debug(f'NO CHANGES were made during reviewing. No retraining needed.')
                self.print_fn(f'[REVIEW] NO CHANGES were made during reviewing. No retraining needed.')
                      
            if not self.finish_function is None:
                self.finish_function()
                      
        assert len(self.suggestions)==0
        highlighter = None
        if self.term_highlighter.trained and how_many<=20:
            highlighter = self.term_highlighter
        text_for_label = [suggestion.get_htmldocview(highlighter=highlighter)
                          for suggestion in self.labeled_data[-how_many:]]

        df = pd.DataFrame(
                       {
                        'example': text_for_label,
                        'changed':[True]*how_many,
                        'label':[HRSystem.RELEVANT_LABEL if item.label==DataItem.REL_LABEL else HRSystem.IRRELEVANT_LABEL  
                                 for item in self.labeled_data[-how_many:] ]
                       }
                      )
        self.annotations = pixt.annotate(
                                         df,
                                         options=[HRSystem.RELEVANT_LABEL, HRSystem.IRRELEVANT_LABEL],
                                         stop_at_last_example=False,
                                         display_fn=html,
                                         include_cancel=False,
                                         final_process_fn=after_reviewing
                                        )
        
        
