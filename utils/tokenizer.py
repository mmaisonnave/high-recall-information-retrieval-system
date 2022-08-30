import string
import spacy




class Tokenizer(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser','ner'])
    def _remove_punctuation(word):
        return ''.join([char for char in word if not char in string.punctuation+' '])
    def _is_valid_word(word):
        return not word.isnumeric() and len(Tokenizer._remove_punctuation(word.replace('\n', '')))!=0 and len(word)>1
    def tokenize(self, str_):
        tokens = [word.lemma_.lower().replace('\n','') for word in self.nlp(str_) if not word.is_stop and word.lemma_.isalnum()]
        # tokens =  # [word.replace('\n', '') for word in tokens if ]
        return filter(Tokenizer._is_valid_word, tokens)
    
    def _bigrams(list_):
        return list(zip(list_[:-1], list_[1:]))
    def _trigrams(list_):
        return list(zip(list_[:-2], list_[1:-1], list_[2:]))
    def ngrams(list_):
        list_=list(list_)
        return Tokenizer._bigrams(list_)+Tokenizer._trigrams(list_)