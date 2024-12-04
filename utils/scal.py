import shutil
import numpy as np
from utils.term_highlighter import TermHighlighter
from utils.data_item import DataItem, QueryDataItem
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
from sklearn.metrics import pairwise_distances
from bs4 import BeautifulSoup
import pickle

from scipy import sparse

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

class SCAL(object):    
    RELEVANT_LABEL='Relevant'
    IRRELEVANT_LABEL='Irrelevant'
    HOME_FOLDER=f'sessions/scal/'
    
    def __init__(self,
                 session_name=None,
                 labeled_collection=None,
                 unlabeled_collection=None,
                 batch_size_cap=10,
                 random_sample_size=10000,
                 target_recall=0.8,
                 simulation=False,
                 empty=False,
#                  proportion_relevance_feedback=1.0,
                 ranking_function='relevance',
                 item_representation=None,
#                  diversity=False,
#                  average_diversity=False,
                 seed=2022
                ):
        if not empty:
            assert not session_name is None , 'If not loading session from disk (not empty SCAL) then a session name is required.'
            self.home_folder = SCAL.HOME_FOLDER
            if simulation:
                self.home_folder = os.path.join(self.home_folder,'simulations/')
            self.home_folder = os.path.join(self.home_folder, session_name)
            
            
            self.initial_label_size=len(labeled_collection)
            
            # CREATE LOG AND SESSION FOLDER
            log_folder_created=False
            session_folder_created=False
            if not os.path.exists(os.path.join(self.home_folder,'log/')):
                log_folder_created=True
                if not os.path.exists(self.home_folder):
                    session_folder_created=True
                    os.mkdir(self.home_folder)
                os.mkdir(os.path.join(self.home_folder,'log/'))
                
            logging.basicConfig(filename=os.path.join(self.home_folder, 'log/scal_system.log'), 
                                format='%(asctime)s [%(levelname)s] %(message)s' ,
    #                             encoding='utf-8',                # INVALID WHEN CHANGE ENV (IMM -> BERT)
                                datefmt='%Y-%m-%d %H:%M:%S',
                                force=True,                      # INVALID WHEN CHANGE ENV (IMM -> BERT)
                                level=logging.DEBUG)
            
            
            logging.debug('-'*30+'STARTING SCAL'+'-'*30)
            logging.debug('Creating new session from scratch (Creating non-empty instance to start SCAL process, NOT loading from disk).')
            if session_folder_created:
                logging.debug(f'Creating session folder: {self.home_folder}')
            if log_folder_created:
                logging.debug(f'Creating log     folder: '+os.path.join(self.home_folder,'log/'))
                
        
#             self.using_relevance_feedback=True
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t session_name=                 {session_name}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t initial_label_size=           {self.initial_label_size}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t len(labeled_collection)=      {len(labeled_collection)}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t len(unlabeled_collection)=    {len(unlabeled_collection)}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t simulation=                   {simulation}')
#             logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t diversity=                    {diversity}')
#             logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t average_diversity=            {average_diversity}')
#             logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t proportion_relevance_feedback={proportion_relevance_feedback}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t target_recall=                {target_recall}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t random_sample_size=           {random_sample_size}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t batch_size_cap=               {batch_size_cap}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t ranking_function=             {ranking_function}')
#             logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t Using relevance feedback=     {self.using_relevance_feedback}')
            if not item_representation is None:
                logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t Doc representation received=  {len(item_representation)} vectors')
                
            self.item_representation=item_representation


            self.simulation=simulation
#             self.diversity=diversity
#             self.average_diversity=average_diversity
#             self.proportion_relevance_feedback = proportion_relevance_feedback
            self.session_name=session_name
            self.target_recall=target_recall
            self.random_sample_size=random_sample_size
            self.ranking_function = ranking_function
            self.seed = seed
            self.ran = np.random.default_rng(self.seed)
            self.B = 1 
            self.n = batch_size_cap
                    
            self.random_unlabeled_collection = self.ran.choice(unlabeled_collection, 
                                                          size=min(self.random_sample_size,len(unlabeled_collection)), 
                                                          replace=False)
            assert all([item.is_unknown() for item in unlabeled_collection])
            
            self.full_U = self.random_unlabeled_collection
            self.cant_iterations = SCAL._cant_iterations(len(self.random_unlabeled_collection))
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t No. of iterations required=   {self.cant_iterations}')
            
#             no_of_synthetic = len([item for item in self.labeled_collection if item.is_synthetic()])
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t Effort (# of labels required)={self._total_effort()}')
            logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t seed (# of labels required)=  {self.seed}')

            self.Rhat=np.zeros(shape=(self.cant_iterations,))
            self.j=0            #-1
            self.labeled_collection = labeled_collection
            self.unlabeled_collection = unlabeled_collection
            self.removed = []
            self.models=[]
            self.precision_estimates=[]
            assert all([item.is_relevant() or item.is_irrelevant() for item in labeled_collection])
            self.all_texts = [item.get_htmldocview() for item in labeled_collection]
            self.all_labels = [SCAL.RELEVANT_LABEL if item.is_relevant() else SCAL.IRRELEVANT_LABEL for item in labeled_collection]
        else:
            logging.debug('Creating empty SCAL system object, parameters should be configured before using.')

    def to_disk(self):
        configuration = {'session-name':self.session_name,
                         'target-recall':self.target_recall,
                         'initial-label-size':self.initial_label_size,
#                          'proportion-relevance-feedback':self.proportion_relevance_feedback,
#                          'using-relevance-feedback':self.using_relevance_feedback,
                         'ranking-function':self.ranking_function,
                         'seed':self.seed,
#                          'diversity':self.diversity,
#                          'average-diversity':self.average_diversity,
                         'simulation':self.simulation,
                         'random-sample-size':self.random_sample_size,
                         'B': self.B,
                         'n': self.n,
                         'b': self.b,
                         'random-unlabeled-collection': [item._dict() for item in self.random_unlabeled_collection],
                         'full-U': [item._dict() for item in self.full_U], 
                         'cant-iterations': self.cant_iterations,
                         'Rhat': list(self.Rhat),
                         'j': self.j,
                         'labeled-collection': [item._dict() for item in self.labeled_collection],
                         'unlabeled-collection': [item._dict() for item in self.unlabeled_collection],
                         'removed': [[elem._dict() for elem in list_] for list_ in self.removed],
#                          'models': self.models,
                         'precision-estimates': self.precision_estimates,
                         'all-texts': self.all_texts,
                         'all-labels': self.all_labels,
                        }

        ####################
        # CREATING FOLDERS #
        ####################
        if not os.path.exists(os.path.join(self.home_folder, 'data/')):
            
#             (f'sessions/scal/{self.session_name}')
            os.mkdir(os.path.join(self.home_folder, 'data/'))
#             if not os.path.exists(f'sessions/scal/{session_name}/log/'):
#                 os.mkdir(f'sessions/scal/{self.session_name}/log')
            os.mkdir(os.path.join(self.home_folder, 'models/'))
            logging.debug(f'Creating data   folder: '+os.path.join(self.home_folder, 'data/'))
            logging.debug(f'Creating models folder: '+os.path.join(self.home_folder, 'models/'))
            
            
        ####################
        #  SAVING MODELS   #
        ####################
        for idx,model in enumerate(self.models):
            model.to_disk(os.path.join(self.home_folder,f'models/model_{idx}'))
        logging.debug(f'Saving {len(self.models)} models into model folder '+os.path.join(self.home_folder, 'models/'))
        
        
        
        ####################
        #  SAVING CONFIG   #
        ####################
        with open(os.path.join(self.home_folder,'data/configuration.json'),'w') as outputfile:
            outputfile.write(json.dumps(configuration, indent=4))
        logging.debug(f'Saving configuration file into data folder         '+os.path.join(self.home_folder, 'data/')) 
                
        ######################
        #  SAVING DOC REPR   #
        ######################  
        if not self.item_representation is None:
            pickle.dump(self.item_representation, open(os.path.join(self.home_folder,'data/item_representation.pickle'), 'wb'))
            logging.debug(f'Saving item_representation file into data folder   '+os.path.join(self.home_folder,'data/item_representation.pickle')) 
            
            
    def from_disk(session_name):        
        home_folder = SCAL.HOME_FOLDER
        if not os.path.exists(os.path.join(home_folder,session_name)):
            # IF session does not exist, look into simulation sessions.
            home_folder = os.path.join(home_folder,'simulations/')
            assert os.path.exists(os.path.join(home_folder,f'simulations/{session_name}')), 'Session not found '\
                                                                                            '(not in regular or simulated sessions).'

        home_folder = os.path.join(home_folder, session_name)
        
        logging.basicConfig(filename=os.path.join(home_folder, f'log/scal_system.log'), 
                            format='%(asctime)s [%(levelname)s] %(message)s' ,
#                             encoding='utf-8',                # INVALID WHEN CHANGE ENV (IMM -> BERT)
                            datefmt='%Y-%m-%d %H:%M:%S',
                            force=True,                      # INVALID WHEN CHANGE ENV (IMM -> BERT)
                            level=logging.DEBUG)
        logging.debug('-'*30+'LOADING  SCAL'+'-'*30)
        
        if 'scal/simulations' in home_folder:
            logging.debug('Session retrieved from SIMULATIONS folder')
        else:
            logging.debug('Session retrieved from regular session folder')
            
        ##############################
        # READING CONFIGURATION FILE #
        ##############################
        with open(os.path.join(home_folder,f'data/configuration.json'), 'r') as f:
            configuration = json.load(f)
            
        logging.debug(f'Loading SCAL system configuration file '+os.path.join(home_folder,f'data/configuration.json'))
        scal = SCAL(empty=True)
        scal.home_folder = home_folder
        
        scal.session_name=configuration['session-name'] #session_name
        
        scal.simulation=configuration['simulation'] #session_name
        scal.initial_label_size=configuration['initial-label-size']
        scal.target_recall=configuration['target-recall']
#         scal.proportion_relevance_feedback = configuration['proportion-relevance-feedback']
#         scal.using_relevance_feedback = configuration['using-relevance-feedback']        
        scal.random_sample_size= configuration['random-sample-size']
        scal.seed=configuration['seed']
        scal.ran =  np.random.default_rng(scal.seed)
        scal.B = configuration['B']
        scal.n = configuration['n']
        scal.b = configuration['b']
        scal.ranking_function = configuration['ranking-function']
#         scal.diversity = configuration['diversity']
#         scal.average_diversity = configuration['average-diversity']
        scal.random_unlabeled_collection = [DataItem.from_dict(dict_) for dict_ in configuration['random-unlabeled-collection']] 
        scal.full_U = [DataItem.from_dict(dict_) for dict_ in configuration['full-U']] 
        scal.cant_iterations =  configuration['cant-iterations']
        scal.Rhat= np.array(configuration['Rhat']) #np.zeros(shape=(self.cant_iterations+1,))
        scal.j=configuration['j']            
        scal.labeled_collection = [DataItem.from_dict(dict_) if 'id' in dict_ else QueryDataItem.from_dict(dict_)  
                                   for dict_ in configuration['labeled-collection']] 
        
        scal.unlabeled_collection = [DataItem.from_dict(dict_) for dict_ in configuration['unlabeled-collection']] 

        scal.removed = [[DataItem.from_dict(dict_) for dict_ in list_] for list_ in configuration['removed']]
#         scal.models=configuration['models']
        scal.precision_estimates=configuration['precision-estimates']
    
    
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t session_name=                 {scal.session_name}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t len(labeled_collection)=      {len(scal.labeled_collection)}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t len(unlabeled_collection)=    {len(scal.unlabeled_collection)}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t simulation=                   {scal.simulation}')
#         logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t diversity=                    {scal.diversity}')
#         logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t average_diversity=            {scal.average_diversity}')
#         logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t proportion_relevance_feedback={scal.proportion_relevance_feedback}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t target_recall=                {scal.target_recall}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t random_sample_size=           {scal.random_sample_size}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t batch_size_cap=               {scal.n}')
#         logging.debug(f'SCAL INITIAL CONFIGURATION.\t\t Using relevance feedback=     {scal.using_relevance_feedback}')
        
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t No. of iterations required=   {scal.cant_iterations}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t Effort (# of labels required)={scal._total_effort()}')
        logging.debug(f'SCAL LOADED  CONFIGURATION.\t\t seed (# of labels required)=  {scal.seed}')
        
        # MODELS
        scal.models=[]
        for idx,model in enumerate([file for file in os.listdir(os.path.join(self.home_folder,'models/')) if 'model_' in file and file.endswith('json')]):
            scal.models.append(TermHighlighter.from_disk(os.path.join(self.home_folder,f'models/{model[:-5]}')))
#                 f'sessions/scal/{session_name}/models/'))
        logging.debug(f'Loading {len(scal.models)} models from disk ('+os.path.join(self.home_folder,f'models/{model[:-5]}')+')')
        
        scal.all_texts = configuration['all-texts']
        scal.all_labels = configuration['all-labels']

        #######################
        #  LOADING DOC REPR   #
        #######################    
        scal.item_representation=None
        if os.path.exists(os.path.join(scal.home_folder,'data/item_representation.pickle')):
            self.item_representation = pickle.load(open(os.path.join(self.home_folder,'data/item_representation.pickle'), 'rb'))
            logging.debug(f'Loading item_representation file into data folder   '+os.path.join(self.home_folder,'data/item_representation.pickle')) 
        
        
        return scal
        # use from_dict in DataItem and change Rhat to numpy
    
#     def _log_status(self):
#         current_b='N/A'
#         current_rhat='N/A'
#         current_precision='N/A'
#         current_Uj_size='  N/A '
#         current_j=f'  -1'
#         if self.j>=0 and self.j<self.cant_iterations:
#             current_b=f'{self.b:3}'
#             current_rhat=f'{self.Rhat[self.j]}'
#             current_precision=f'{self.precision_estimates[-1]:4.3f}'
#             current_j=f'{self.j:>4}'
#             current_Uj_size=f'{self.size_of_Uj:6}'
#         logging.debug(f'j={current_j}/{self.cant_iterations:4} - B={self.B:<5,} - b={current_b} - Rhat={current_rhat} -'\
#               f' len_unlabeled= {len(self.random_unlabeled_collection):6,} - len_labeled={len(self.labeled_collection):6,}'\
#               f' - cant_rel={self._relevant_count()} - precision={current_precision} - Uj_size={current_Uj_size}'
#              ) 
#         print(f'j={current_j}/{self.cant_iterations:4} - B={self.B:<5,} - b={current_b} - Rhat={current_rhat} -'\
#               f' len_unlabeled= {len(self.random_unlabeled_collection):6,} - len_labeled={len(self.labeled_collection):6,}'\
#               f' - cant_rel={self._relevant_count()} - precision={current_precision} - Uj_size={current_Uj_size}'
#              ) 
#     def resume(self):
#         pass

    def run(self):
#         self.j=0    # WRONG
        if self.j<self.cant_iterations:
            # ONLY RUN IF THERE IS ANY ITERATION LEFT TO DO. 
            logging.debug('(RE)STARTING SCAL algorithm')
            self.loop()
        else:
            logging.debug('Attempt to run an already finished SCAL process. Skipping ...')
            print('SCAL PROCESS FINISHED. Nothing to do skipping. ')
        
    def _extend_with_random_documents(self):    
        assert all([item.is_unknown() for item in self.random_unlabeled_collection]), f'len(random_unlabeled_collection)={len(self.random_unlabeled_collection)}'
        extension = self.ran.choice(self.random_unlabeled_collection, size=min(100,len(self.random_unlabeled_collection)), replace=False)
        list(map(lambda x: x.set_irrelevant(), extension))
        assert all([item.is_irrelevant() for item in extension])
        return extension
    
    def _label_as_unknown(collection):
        list(map(lambda x: x.set_unknown(), collection))
        
    def _build_classifier(self, training_collection):
        model = TermHighlighter()
        model.fit(training_collection, item_representation=self.item_representation)
        return model
    
    def _smallest_distance_to_labeled_collection(self, aggregation ):
        assert aggregation=='average' or aggregation=='min', 'Please specify a valid aggregation function (average or min).'
        item_list = self.random_unlabeled_collection
        if not self.item_representation is None:
            if type(self.item_representation[list(self.item_representation)[0]])==np.ndarray:
                m1 = np.vstack([self.item_representation[item.id_] for item in item_list])
                m2 = np.vstack([self.item_representation[item.id_] for item in self.labeled_collection])
            else:
                m1 = sparse.vstack([self.item_representation[item.id_] for item in item_list])
                m2 = sparse.vstack([self.item_representation[item.id_] for item in self.labeled_collection])
        else:

            m1 = DataItem.get_X(item_list)
            m2 = DataItem.get_X(self.labeled_collection)
        
        
        distances = pairwise_distances(m1,m2)
        if aggregation=='average':
            mindist = np.average(distances,axis=1)
        elif  aggregation=='min':
            mindist = np.min(distances,axis=1)
        
            
        if np.max(mindist)!=0:
            mindist=mindist/np.max(mindist)
        assert mindist.shape==(len(self.random_unlabeled_collection),)
        return mindist
    
        
    def _select_highest_scoring_docs(self, function):
        """
            valid functions: "relevance" "uncertanty" "avg_distance" "min_distance"
        """
        # RELEVANCE 
        if function=='relevance':
            yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
            args = np.argsort(yhat)[::-1]
            
            
        elif function=='relevance_with_avg_diversity':
            yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
            mindist = self._smallest_distance_to_labeled_collection(aggregation='average')
            haverage = 2*((mindist*yhat)/(mindist+yhat))
            assert mindist.shape==yhat.shape, f'{mindist.shape}!={yhat.shape}'
            args = np.argsort(haverage)[::-1]
        elif function=='relevance_with_min_diversity':
            yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
            mindist = self._smallest_distance_to_labeled_collection(aggregation='min')
            haverage = 2*((mindist*yhat)/(mindist+yhat))
            assert mindist.shape==yhat.shape, f'{mindist.shape}!={yhat.shape}'
            args = np.argsort(haverage)[::-1]
            
        elif function=='half_relevance_half_uncertainty':
            current_proportion = len(self.labeled_collection)/(self._total_effort()+1)
            if current_proportion<0.5:
                yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(yhat)[::-1]
            else:
                yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(np.abs(yhat-0.5))
                
        elif function=='1quarter_relevance_3quarters_uncertainty':
            current_proportion = len(self.labeled_collection)/(self._total_effort()+1)
            if current_proportion<0.25:
                yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(yhat)[::-1]
            else:
                yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(np.abs(yhat-0.5))
                
        elif function=='random':
            args = list(range(len(self.random_unlabeled_collection)))
            np.random.shuffle(args)
                        
        elif function=='3quarter_relevance_1quarters_uncertainty':
            current_proportion = len(self.labeled_collection)/(self._total_effort()+1)
            if current_proportion<0.75:
                yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(yhat)[::-1]
            else:
                yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(np.abs(yhat-0.5))
            
        elif function=='avg_distance':
            mindist = self._smallest_distance_to_labeled_collection(aggregation='average')
            args = np.argsort(mindist)[::-1]
        elif function=='min_distance':
            mindist = self._smallest_distance_to_labeled_collection(aggregation='min')
            args = np.argsort(mindist)[::-1]
        
        # UNCERTAINTY
        elif function=='uncertainty':
            yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
            args = np.argsort(np.abs(yhat-0.5))
        elif function=='uncertainty_with_avg_diversity':
            yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
            auxiliar = 1/(1+np.abs(yhat-0.5))  # Higher when most uncertainty   [0.66, 0.67, ..., 0.98, ..., 0.67, 0.66]
            mindist = self._smallest_distance_to_labeled_collection(aggregation='average')
            haverage = 2*((mindist*auxiliar)/(mindist+auxiliar))
            args = np.argsort(haverage)[::-1]
        elif function=='uncertainty_with_min_diversity':
            yhat = self.models[-1].predict(self.random_unlabeled_collection, item_representation=self.item_representation)
            auxiliar = 1/(1+np.abs(yhat-0.5))  # Higher when most uncertainty   [0.66, 0.67, ..., 0.98, ..., 0.67, 0.66]
            mindist = self._smallest_distance_to_labeled_collection(aggregation='min')
            haverage = 2*((mindist*auxiliar)/(mindist+auxiliar))
            args = np.argsort(haverage)[::-1]
        else:
            assert False, f'Invalid ranking function: {function}'

        return [self.random_unlabeled_collection[arg] for arg in args[:self.B]]
            
        
#         current_proportion = len(self.labeled_collection)/(self._total_effort()+1)
#         yhat = self.models[-1].predict(self.random_unlabeled_collection)
#         if current_proportion<=self.proportion_relevance_feedback:
# #             logging.debug(f'{current_proportion} <= {self.proportion_relevance_feedback}? TRUE')
#             # RELEVANCE SAMPLING
#             if self.diversity:
#                 # WITH DIVERSITY
#                 mindist = self._smallest_distance_to_labeled_collection()
#                 haverage = 2*((mindist*yhat)/(mindist+yhat))
#                 assert mindist.shape==yhat.shape, f'{mindist.shape}!={yhat.shape}'
#                 args = np.argsort(haverage)[::-1]
#             else:
#                 # WITHOUT DIVERSITY
#                 args = np.argsort(yhat)[::-1]
# #             highest_scoring_docs = [self.random_unlabeled_collection[arg] for arg in args[:self.B]]
            
# #             return 
#         else:
# #             logging.debug(f'{current_proportion} <= {self.proportion_relevance_feedback}? FALSE')
#             if self.using_relevance_feedback:
#                 logging.debug('Change from relevance sampling to uncertainty sampling')
#                 self.using_relevance_feedback=False
#             # UNCERTAINTY SAMPLING
#             if self.diversity:
#                 # WITH DIVERSITY
#                 mindist = self._smallest_distance_to_labeled_collection()
#                 assert mindist.shape==yhat.shape, f'{mindist.shape}!={yhat.shape}'
#                 auxiliar = 1/(1+np.abs(yhat-0.5))
#                 if np.max(auxiliar)!=0:
#                     auxiliar=auxiliar/np.max(auxiliar)
#                 haverage = 2*((mindist*auxiliar)/(mindist+auxiliar))
                
#                 args = np.argsort(haverage)[::-1]
#             else:
#                 # WITHOUT DIVERSITY
#                 args = np.argsort(np.abs(yhat-0.5))
            
#         return [self.random_unlabeled_collection[arg] for arg in args[:self.B]]
        
    def _remove_from_unlabeled(self,to_remove):
        to_remove = set(to_remove)
        return list(filter(lambda x: not x in to_remove, self.random_unlabeled_collection))

    
    def _get_Uj(self,j):
        to_remove = set([elem for list_ in self.removed[:(j+1)]  for elem in list_])
        Uj = [elem for elem in self.full_U if not elem in to_remove]
        return Uj
    def _total_effort(self):  
#         if not hasattr(self, 'labeling_budget'):
        B=1
        it=1
        effort=0
#             len_unlabeled=self._unlabeled_in_sample()
        len_unlabeled=self.random_sample_size
        while (len_unlabeled>0):        
            b = B if B<=self.n else self.n
            effort+=min(b,len_unlabeled)
            len_unlabeled = len_unlabeled - B
            B+=int(np.ceil(B/10))
            it+=1
        self.labeling_budget = effort
        
        return self.labeling_budget   
    def _cant_iterations(len_unlabeled):    
        B=1
        it=0
        while len_unlabeled>0:        
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
        effort = len(self.labeled_collection) - self.initial_label_size
#         no_of_synthetic = len([item for item in self.labeled_collection if item.is_synthetic()])
        total_effort = self._total_effort() # +1 for the topic description (which is included in the labeled collection)
        str_=f'{int(100*(effort/total_effort)):3} %'
        # print(f'{int(effort/total_effort):3}', end='')\
        str_+=' |'
        end_=f'| {effort:4}/{total_effort:4}'
        str_+=int((size-(len(str_)+len(end_)))*(effort/total_effort))*'='
        str_+= '-' if (effort%total_effort)!=0 else ''
        print(str_+' '*(size-len(str_)-len(end_)) +end_)
        
        
    def _show_ids(item_list):
        if len(item_list)==0:
            return '<>'
        elif len(item_list)==1:
            return f'<{item_list[0].id_}({item_list[0].label})>'
        elif len(item_list)==2:
            return f'<{item_list[0].id_}({item_list[0].label}), {item_list[1].id_}({item_list[1].label})>'
        elif len(item_list)==3:
            return f'<{item_list[0].id_}({item_list[0].label}), {item_list[1].id_}({item_list[1].label}), {item_list[2].id_}({item_list[2].label})>'
        elif len(item_list)==4:
            return f'<{item_list[0].id_}({item_list[0].label}), {item_list[1].id_}({item_list[1].label}), {item_list[2].id_}({item_list[2].label}), {item_list[3].id_}({item_list[3].label})>'
        else:
            return f'<{item_list[0].id_}({item_list[0].label}), {item_list[1].id_}({item_list[1].label}), ..., {item_list[-2].id_}({item_list[-2].label}), {item_list[-1].id_}({item_list[-1].label})>'
            
    def loop(self):
        """
        Executes the main iterative loop for the labeling and model training process.
        In every iteration, all data items are presented to the user again, including previously labeled 
        items and new suggestions for labeling. Previously labeled items are shown with their user-assigned 
        labels, while suggestions display the model-predicted labels.

        The system starts by showing the first suggestion. Moving backward lets the user revisit and correct 
        previously reviewed items, while moving forward displays the next suggestions. During each iteration, 
        users can update labels they have already assigned or correct the model's suggested labels.

        Once the updated labels are stored, the next loop trains a new model using the latest labeled data. 
        This includes the new suggestions with user-corrected labels and previously labeled items that may 
        have been adjusted.


    
        This method performs the following steps:
        1. Extends the labeled collection with a random selection of documents.
        2. Trains a new classification model based on the extended collection.
        3. Identifies the most relevant documents from the unlabeled pool using the ranking function.
        4. Samples a batch of documents for further annotation.
        5. Displays the selected documents to the user (if not in simulation mode) 
        and collects annotations for the batch.
        6. Updates the annotations and continues the process until all iterations are completed.

        Attributes updated:
            - `self.models`: Appends the newly trained model.
            - `self.sorted_docs`: Stores documents sorted by relevance.
            - `self.random_sample_from_batch`: Contains the randomly sampled batch for annotation.
            - `self.all_texts`: Updates the list of all document texts displayed.
            - `self.annotations`: Stores the user's annotations (in non-simulation mode).

        Note:
            In simulation mode, annotations are skipped, and the process transitions directly
            to the next iteration.

        The storage of all models (not just the last one) and the tracking of Rhat, r, and 
        other variables are required to compute the threshold for the final classifier in a 
        manner that ensures a specific level of recall.
        """
        # STATUS BAR
        if not self.simulation:
            print('-'*109)
            print(f'Session name:       {self.session_name:50}  Total size of database: {len(self.unlabeled_collection):,}')
            topic_description=BeautifulSoup(self.labeled_collection[0].get_htmldocview(),'html.parser').get_text()
            if self.labeled_collection[0].is_synthetic():
                print(f"Topic description:  '{topic_description[:min(len(topic_description),80)]}'")
            print('- '*54+'-')
            print(f'Labeled documents: {len(self.labeled_collection)} '\
                  f'({self._relevant_count():8,} relevant / {self._irrelevant_count():8,} irrelevants)\t\t'\
                  f' Unlabeled documents: {self._unlabeled_in_sample():8,}')
            self._progress_bar(109)
            print('-'*109)
            # ~
        
        # LOG iteration no. X started
        logging.debug(f' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LOOP started IT # {self.j+1} (j={self.j}) ')
        self.b = self.B if (self.Rhat[self.j]==1 or self.B<=self.n) else self.n
        assert self.b<=self.B
        logging.debug(f'IN-LOOP. Labeling batch =       {self.b}.')
        precision = f'{self.precision_estimates[-1]:4.3f}' if len(self.precision_estimates)>0 else 'N/A'
        
        # LOG batch size min(x,cap)=y
        
        
        extension = self._extend_with_random_documents()
        logging.debug(f'IN-LOOP. Extension with random {len(extension)} documents (<{extension[0].id_}({extension[0].label}), ..., {extension[-1].id_}({extension[-1].label})>).')
        

        # new model created, 
        self.models.append(self._build_classifier(list(extension)+list(self.labeled_collection)))
        logging.debug(f'IN-LOOP. New model trained with {len(list(extension)+list(self.labeled_collection))} documents '\
                      f'({SCAL._show_ids(list(extension)+list(self.labeled_collection))}).')

        SCAL._label_as_unknown(extension)
        logging.debug('Looking for suggestions in random sample of unlabeled documents ...')
        self.sorted_docs = self._select_highest_scoring_docs(function=self.ranking_function)
        logging.debug(f'IN-LOOP. Looking for            {len(self.sorted_docs)} most relevant documents '\
                f'({SCAL._show_ids(self.sorted_docs)})')

        self.random_sample_from_batch = self.ran.choice(self.sorted_docs, size=self.b, replace=False)
                  
        logging.debug(f'IN-LOOP. Random sample          {len(self.random_sample_from_batch)} documents from suggested relevant '\
                f'({SCAL._show_ids(self.random_sample_from_batch)})')
                  
        logging.debug('Computing scores to be used as highlighting ...')   
        yhat = self.models[-1].predict(self.random_sample_from_batch, item_representation=self.item_representation)
        
        highlighter = None
        if not self.simulation:
            highlighter = self.models[-1]            
                  
        text_for_label = [suggestion.get_htmldocview(highlighter=highlighter, confidence_score=confidence)
                          for suggestion,confidence in zip(self.random_sample_from_batch,yhat)]
        
        client_current_index = len(self.all_texts)+1
        self.all_texts += text_for_label
        logging.debug('Displaying suggestions to the user ...')
        df = pd.DataFrame(
                       {
                        'example': self.all_texts,
                        'changed':[False]*len(self.all_texts),
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
    """
    Performs post-iteration updates and bookkeeping after each loop in the SCAL process.  

    The bookkeeping tracks labeled and unlabeled items, the model’s performance to date (such as Rhat, precision), 
    and other estimations needed to compute the final threshold.  

    Choosing the right threshold for the final classifier ensures high recall, meeting our requirements.  

    This method performs the following actions:
    - Updates the labeled collection and validates label consistency.
    - Handles label updates based on user-provided annotations or simulation results.
    - Updates the unlabeled collection by removing labeled items and logging changes.
    - Tracks and logs metrics such as precision, true positives, and relevant articles found.
    - Adjusts the threshold (`Tj`) for the classifier and updates related variables.
    - Logs the current state of the SCAL system, including statistics for labeled and unlabeled collections.
    - Prepares for the next iteration or finalizes the process if no more unlabeled data remains.

    Key variables updated:
    - `all_labels`: Stores the updated labels for all items in the labeled collection.
    - `labeled_collection`: Expands with newly labeled data items.
    - `random_unlabeled_collection`: Updates by removing labeled items from the pool of unlabeled data.
    - `precision_estimates`: Tracks precision metrics for each iteration.
    - `Rhat`: Tracks the estimated number of relevant articles found.
    - `B`: Adjusts the batch size for the next iteration.

    Logging and Debugging:
    - Logs various metrics, including precision, true positives, and relevant articles identified.
    - Outputs detailed summaries of the labeled and unlabeled collections after each iteration.

    If more unlabeled data remains, this method triggers the next iteration of the loop. Otherwise, 
    it calls the `finish` method to finalize the SCAL process.

    Raises:
    - AssertionError: If label consistency checks fail.
    """
        if not self.simulation:
            new_labels =  list(self.annotations["label"])
            assert len(new_labels[:-self.b]) == len(self.all_labels)
            count=0
            for old, new in zip(self.all_labels, new_labels[:-self.b]):
                  if old!=new:
                      logging.debug(f'WARNING. Label changed for item id={self.labeled_collection[count].id_} from {old} to {new} '\
                                    f'NEW LABEL: ({self.labeled_collection[count].id_},{new})')
                  count+=1
                  
            
            self.all_labels = list(self.annotations["label"])
        else:
                  
            logging.debug('AFTER LOOP. Label information extracted from Oracle.') 
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
        new_labels_str = [f'({item.id_},{label})' for label,item in zip(self.all_labels[-self.b:], self.random_sample_from_batch)]
        assert self.b==len(self.random_sample_from_batch)
        logging.info(f'AFTER LOOP. NEW LABELS (#{self.b:3}).       ' + ';'.join(new_labels_str))
        logging.info(f'AFTER LOOP. PRECISION:                      {r/self.b} ')
        logging.info(f'AFTER LOOP. True Positives:                 {r} ')
        logging.info(f'AFTER LOOP. False Positives:                {self.b-r} ')
        logging.info(f'AFTER LOOP. NO. OF RELEVANT ARTICLES FOUND: {r} ')
                
                  
        logging.info(f'AFTER LOOP. New labeled collection {len(self.labeled_collection)} documents '\
                      f'({SCAL._show_ids(list(self.labeled_collection))}).')  
                  
        logging.info(f'AFTER LOOP. New unlabeled collection (random sample) {len(self.random_sample_from_batch)} documents '\
          f'({SCAL._show_ids(list(self.random_sample_from_batch))}).')
                  
        Uj = [elem for elem in self.full_U if not elem in set([elem for list_ in self.removed for elem in list_])]
        
        tj = np.min(self.models[self.j].predict([elem for elem in self.full_U if not elem in Uj], item_representation=self.item_representation))
        logging.info(f'Tj (current threshold)={tj}')
        self.size_of_Uj = len(Uj)
        self.precision_estimates.append(r/self.b)
        self.Rhat[self.j] = (r*self.B)/self.b
        assert (r*self.B)/self.b>=r
        if self.j-1>=0:
            self.Rhat[self.j] += self.Rhat[self.j-1]
        
        self.B += int(np.ceil(self.B/10))
        self.B = min(self.B, len(self.random_unlabeled_collection))

        logging.info(f'SCAL IT LOG. it={self.j+1:>4}/{self.cant_iterations:4} - B={self.B:<5,} - b={self.b:3} - Rhat={self.Rhat[self.j]}'\
              f' - len_unlabeled= {len(self.random_unlabeled_collection):6,} - len_labeled={len(self.labeled_collection):6,}'\
              f' - cant_rel={self._relevant_count()} - precision={self.precision_estimates[-1]:4.3f} - Uj_size={self.size_of_Uj:6}'
             ) 
        self.j+=1
        self.to_disk()

        
        if len(self.random_unlabeled_collection)>0:
            if not self.simulation:
                clear_output(wait=False)
            self.loop()
        else:
            self.finish()


    def finish(self):
    """
    Finalizes the SCAL process by completing the following steps:

    1. **Metrics Calculation and Threshold Selection**:
    - Estimates prevalence based on the current model and labeled data.
    - Determines the expected number of relevant items in the dataset using target recall.
    - Identifies the iteration where the desired recall is achieved and calculates the classification threshold for 
      final predictions.

    2. **Labeled Data Export**:
    - Saves all labeled data to a file for further analysis.
    - Logs the labeled data statistics, including counts of relevant and irrelevant articles.

    3. **Final Model Training**:
    - Builds a final classifier using the complete labeled dataset to improve prediction accuracy.

    4. **Unlabeled Data Prediction**:
    - Applies the final model to the remaining unlabeled data to identify additional potentially relevant 
      items using the identify threshold to achieve the desire recall. 
    - Exports predicted relevant items with confidence scores.

    5. **Simulation and Performance Metrics** (if applicable):
    - Evaluates model performance (accuracy, precision, recall, F1-score) against ground truth in a simulation 
      environment.
    - Logs detailed simulation results, including confusion matrix components.

    6. **File Export and Cloud Upload**:
    - Prepares the final output file for export, containing relevant and suggested articles with their 
      confidence scores.
    - Optionally uploads the file to a specified cloud storage location if not in simulation mode.

    Returns:
    - A list of relevant data items and their associated confidence scores.

    Raises:
    - AssertionError: If file preparation or exporting fails.
    """


        print('Preparing for exporting ... (this could take around 3 hours)')
        logging.debug(f'SCAL PROCESS FINISHED. Process finished. Number of labeled articles={len(self.labeled_collection)}'\
                                           f' Number of unlabeled articles in the random sample={len(self.random_unlabeled_collection)}')
                  

#         logging.debug(f'it=  end    - B={self.B:<5,} - b={self.b:2} - Rhat={self.Rhat[self.j-1]:8.3f} -'\
#               f' len_unlabeled= {len(self.random_unlabeled_collection):6,} - len_labeled={len(self.labeled_collection):6,}'\
#               f' - cant_rel={self._relevant_count()} - precision={self.precision_estimates[-1]:4.3f}'
#              )
#         print(f'Rhat={self.Rhat}')
        self.prevalecence = (1.05*self.Rhat[self.j-1]) / self.random_sample_size
#         print(f'Prevalecense: {self.prevalecence}')
        
        no_of_expected_relevant = self.target_recall * self.prevalecence * self.random_sample_size
#         print(f'no_of_expected_relevant={no_of_expected_relevant}')
        j=0
        while j<len(self.Rhat) and self.Rhat[j]<no_of_expected_relevant:
#             print(f'{self.Rhat[j]} (Rhat) <{no_of_expected_relevant} (no_exp_rel)? {self.Rhat[j]<no_of_expected_relevant}')
            j+=1
            

            
#         print(f'j value: {j} (it={j+1})')
#         Uj = full_U
#         to_remove = set([elem for list_ in self.removed[:j]  for elem in list_])
#         print(f'size of removed: {len(self.removed)}')
#         print(f'size of to_remove: {len(to_remove)}')
#         Uj = [elem for elem in Uj if not elem in to_remove]
        

        
        #t = np.max(self.models[j].predict(Uj)) ANTERIOR QUE POR ALGUNA RAZON ANDABA BIEN
#         t = np.max(self.models[j].predict([elem for elem in self.full_U if not elem in Uj])) # (U_0) \ (U_j) (set difference)
        
        Uj = self._get_Uj(j)
#         print(f'Size of U0: {len(self._get_Uj(0))}')
#         print(f'Size of Uj: {len(Uj)}')       
        
        # By the j-th iteration all the relevant articles (plus some irrelevants) have been found  which means:
        #
        # Rhat_{j} > expected_no_of_rel_articles_in_sample
        # 
        # Here expected_no_of_rel_articles_in_sample =0.8*prevalence*N   (0.8 is target_recall) (and N is sample size)
        # Because by j-th iteration every relevant article was found, we have in Uj mostly irrelevants. So,
        # we need a threshold that mark as relevant at least what we have procesed so far (U_inicial \ Uj) (everything except Uj).
        # to do that, we have to options. Define the threshold t as:
        #     1. t = np.max(self.models[j].predict(Uj)        # This would give a threshold that leaves all elements in Uj as irrelevanats
        #     2. t = np.min(self.models[j].predict(U_ini\Uj)  # This would give the minimum threshold to leave all 
        #                                                                               elements procesed so far as relevant
        #
        # (U_ini\Uj is computed as [elem for elem in self.full_U if not elem in Uj])
        #
        # Both thresholds give almost the same number (one considers the element right before the relevants start and the other the one 
        # right after), which means that the classifier should be implemented as:
        #
        # relevant if self.models[-1].predict(...) >  t     (for first  threshold)
        # relevant if self.models[-1].predict(...) >= t     (for second threshold)
        #         
        
        t = np.min(self.models[j].predict([elem for elem in self.full_U if not elem in Uj], item_representation=self.item_representation))
        
#         print('DEBUG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#         print(f'max(Sj(Uj))        ={np.max(self.models[j].predict(self._get_Uj(j))):4.3f} (mejor)')
#         print(f'max(Sj(U0))        ={np.max(self.models[j].predict(self._get_Uj(0))):4.3f}')
#         print(f'max(Sj(U-1 \\ Uj ))={np.max(self.models[j].predict([elem for elem in self.full_U if not elem in Uj])):4.3f} (propuesta - pero anda mal)')
#         print(f'min(Sj(U-1 \\ Uj ))={np.min(self.models[j].predict([elem for elem in self.full_U if not elem in Uj])):4.3f} (muy parecido al mejor)')
#         print('DEBUG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
#         logging.debug(f'threshold={t} - '\
#                       f'size of U0={len(self._get_Uj(0))} - '\
#                       f'size of Uj={len(Uj)} - '\
#                       f'size of Uo\\Uj={len([elem for elem in self.full_U if not elem in Uj])}')
#         print(f'threshold={t}')
                  
        # SAVING LABELED DATA
        logging.debug(f'Saving {len(self.labeled_collection)} documents as labeled data in '\
                      ''+os.path.join(self.home_folder, f'data/labeled_data')+time.strftime("%Y-%m-%d_%H-%M")+'.csv'\
                      ' before making final predictions')
                  
        with open(os.path.join(self.home_folder, f'data/labeled_data'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'), 'w') as writer:
                  writer.write('\n'.join([';'.join([item.id_,item.label]) for item in self.labeled_collection]))
                  
        # FINAL CLASSIFIER
        logging.debug('Creating final classifier ...')
        self.models.append(self._build_classifier(self.labeled_collection))
#         print(f'Rhat={self.Rhat}')
#         print(f'Size of unlabeled={len(self.unlabeled_collection)}')
#         print(f'Size of labeled={len(self.labeled_collection)}')

        ## CORREGIR
    
        no_of_synthetic = len([item for item in self.labeled_collection if item.is_synthetic()])
        logging.debug(f'Removing {len(self.labeled_collection[no_of_synthetic:])} labeled documents from full unlabeled collection.')
        final_unlabeled_collection = [item for item in self.unlabeled_collection if not item in self.labeled_collection[no_of_synthetic:]]
#         print(f'Size of unlabeled after removing labeled={len(final_unlabeled_collection)}')

        logging.debug('-'*30+'FINISHING SCAL'+'-'*30)
        logging.info(f'Final   labeled size   ={len(self.labeled_collection)}')
        logging.info(f'Final unlabeled size   ={len(final_unlabeled_collection)}')
        logging.info(f'Est. prevalecence      ={(self.prevalecence):5.4f}')
        logging.info(f'Exp. relevant in sample={no_of_expected_relevant:,}')
        logging.info(f'j                      ={j}')
        logging.info(f'Threshold              ={t}')

#         print(f'proportion={self.proportion_relevance_feedback}')
        logging.debug(f'Making prediction over set of unlabeled articles ({len(final_unlabeled_collection):,}).')
        yhat = self.models[-1].predict(final_unlabeled_collection, progress_bar=True, item_representation=self.item_representation)
        
        relevant = yhat>=t      
        logging.info(f'Relevant found (total) ={len([item for item in self.labeled_collection if item.is_relevant()])+np.sum(relevant)}'\
                      f'({len([item for item in self.labeled_collection if item.is_relevant()])} labeled / {np.sum(relevant)} suggested)')
                  
        
        logging.info(f'Found {np.sum(relevant)} possible relevant articles.')
#         print(f'Shape of yhat={yhat.shape}')
#         print(f'Number of relevant={np.sum(yhat>=t)}')
        
        


#         print(f'Relevant count: {np.sum(relevant)}')
#         print(f'Precision estimate: {np.average(self.precision_estimates)}')
        no_of_synthetic = len([item for item in self.labeled_collection if item.is_synthetic()])
        relevant_data = [item for item in self.labeled_collection[no_of_synthetic:] if item.is_relevant()]
        confidence = [1.0]*len(relevant_data)
        
        no_of_labeled_rel = len(relevant_data)
        
        relevant_data += [item for item,y in zip(final_unlabeled_collection,yhat) if y>=t]
        confidence +=list([y for item,y in zip(final_unlabeled_collection,yhat) if y>=t])
        
        assert len(relevant_data)==len(confidence)
        
#         print(f'len(relevant_data)={len(relevant_data)}')
#         print(f'len(confidence)   ={len(confidence)}')
        logging.debug('Preparing file for exporting ...')
        filename = os.path.join(self.home_folder, f'data/exported_data_'+time.strftime("%Y-%m-%d_%H-%M")+'.csv')
        with open(filename, 'w') as writer:
            writer.write('URL,relevant_or_suggested,confidence\n')
            count=0
            for item,confidence_value in zip(relevant_data,confidence):
                if count<no_of_labeled_rel:
                    writer.write(f'https://proquest.com/docview/{item.id_},rel,{confidence_value:4.3f}\n')  
                else:
                    writer.write(f'https://proquest.com/docview/{item.id_},sugg,{confidence_value:4.3f}\n')  
                count+=1
                
                      
        if not self.simulation:
            assert os.path.isfile(filename)
            temp_file ='exported_data_'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'
            shutil.copyfile(filename, temp_file)
            assert os.path.isfile(temp_file)
            os.system(f'aws s3 cp {temp_file} s3://pq-tdm-studio-results/tdm-ale-data/623/results/')
            os.remove(temp_file)
            logging.debug(f'File {temp_file} sent over email.')
            
        if self.simulation:
            ytrue = [1 if Oracle.is_relevant(item) else 0 for item in final_unlabeled_collection]
            acc = accuracy_score(ytrue,yhat>=t)
            prec = precision_score(ytrue, yhat>=t)
            rec = recall_score(ytrue, yhat>=t)
            f1 = f1_score(ytrue, yhat>=t)
            tn, fp, fn, tp = confusion_matrix(ytrue, yhat>=t).ravel()
            logging.debug(f'SIMULATION RESULTS: accuracy={acc:4.3f} - precision={prec:4.3f} - recall={rec:4.3f} - F1-score={f1:4.3f} - '\
                          f'TN={tn} - FP={fp} - FN={fn} - TP={tp} ')
#             print(f'SIMULATION RESULTS: accuracy={acc:4.3f} - precision={prec:4.3f} - recall={rec:4.3f} - F1-score={f1:4.3f} - '\
#                           f'TN={tn} - FP={fp} - FN={fn} - TP={tp} ')
#             print(f'Accuracy:  {acc:4.3f}')
#             print(f'Precision: {prec:4.3f}')
#             print(f'Recall:    {rec:4.3f}')
#             print(f'F1-score   {f1:4.3f}')
            print('FINISH simulation')
        return relevant_data, confidence
#         return _get_relevant(model, prevalecence, unlabeled_collection)
        
