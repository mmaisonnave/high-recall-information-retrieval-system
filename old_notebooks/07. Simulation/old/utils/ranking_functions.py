from utils.data_item import DataItem
from sklearn.metrics import pairwise_distances

def relevance_sampling(unlabeled_collection, predictive_model, ):
    yhat = predictive_model.predict(unlabeled_collection)
    args = np.argsort(yhat)[::-1]
    return [unlabeled_collection[arg] for arg in args[:self.B]]
    
    
def uncertanty_sampling(unlabeled_collection, predictive_model, ):
    yhat = predictive_model.predict(unlabeled_collection)
    args = np.argsort(np.abs(yhat-0.5))    
    return [unlabeled_collection[arg] for arg in args[:self.B]]
    
    
    
    
# WITH DIVERSITY
def _smallest_distance_to_labeled_collection(unlabeled_collection, labeled_collection, type_): 
    """
        valid types = [average, min]
    """
    m1 = DataItem.get_X(unlabeled_collection)
    m2 = DataItem.get_X(labeled_collection)
    distances = pairwise_distances(m1,m2)
    if type_=='average':
        mindist = np.average(distances,axis=1)
    elif type_=='min':
        mindist = np.min(distances,axis=1)

    if np.max(mindist)!=0:
        mindist=mindist/np.max(mindist)    
    return mindist


def relevance_sampling_with_average_diversity(unlabeled_collection, predictive_model, labeled_collection, type_):
    yhat = predictive_model.predict(unlabeled_collection)
    
    mindist = self._smallest_distance_to_labeled_collection(unlabeled_collection, labeled_collection,type_)
    haverage = 2*((mindist*yhat)/(mindist+yhat))
    assert mindist.shape==yhat.shape, f'{mindist.shape}!={yhat.shape}'
    args = np.argsort(haverage)[::-1]
    return [unlabeled_collection[arg] for arg in args[:self.B]]

def relevance_sampling(unlabeled_collection, predictive_model, ):
        current_proportion = len(self.labeled_collection)/(self._total_effort()+1)
        yhat = self.models[-1].predict(self.random_unlabeled_collection)
        if current_proportion<=self.proportion_relevance_feedback:
#             logging.debug(f'{current_proportion} <= {self.proportion_relevance_feedback}? TRUE')
            # RELEVANCE SAMPLING
            if self.diversity:
                # WITH DIVERSITY
                mindist = self._smallest_distance_to_labeled_collection()
                haverage = 2*((mindist*yhat)/(mindist+yhat))
                assert mindist.shape==yhat.shape, f'{mindist.shape}!={yhat.shape}'
                args = np.argsort(haverage)[::-1]
            else:
                # WITHOUT DIVERSITY
                args = np.argsort(yhat)[::-1]
#             highest_scoring_docs = [self.random_unlabeled_collection[arg] for arg in args[:self.B]]
            
#             return 
        else:
#             logging.debug(f'{current_proportion} <= {self.proportion_relevance_feedback}? FALSE')
            if self.using_relevance_feedback:
                logging.debug('Change from relevance sampling to uncertainty sampling')
                self.using_relevance_feedback=False
            # UNCERTAINTY SAMPLING
            if self.diversity:
                # WITH DIVERSITY
                mindist = self._smallest_distance_to_labeled_collection()
                assert mindist.shape==yhat.shape, f'{mindist.shape}!={yhat.shape}'
                auxiliar = 1/(1+np.abs(yhat-0.5))
                if np.max(auxiliar)!=0:
                    auxiliar=auxiliar/np.max(auxiliar)
                haverage = 2*((mindist*auxiliar)/(mindist+auxiliar))
                
                args = np.argsort(haverage)[::-1]
            else:
                # WITHOUT DIVERSITY
                args = np.argsort(np.abs(yhat-0.5))
            
        return [self.random_unlabeled_collection[arg] for arg in args[:self.B]]