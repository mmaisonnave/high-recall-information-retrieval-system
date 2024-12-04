"""
This module provides a `Tokenizer` class for processing and extracting meaningful tokens and n-grams 
from text. It leverages the SpaCy NLP library for linguistic processing and includes utility methods 
to handle punctuation, validate tokens, and calculate n-gram indices.

Classes:
--------
Tokenizer : 
    A class for tokenizing text, filtering valid tokens, and generating n-grams with optional index 
    information.

Functions:
----------
Tokenizer._remove_punctuation(word):
    Removes punctuation and whitespace from a given word.

Tokenizer._is_valid_word(word):
    Checks if a word is valid (not numeric, has no punctuation, and is longer than one character).

Tokenizer.tokenize(str_, include_index=False):
    Tokenizes the input string into lemmatized, lowercased tokens, excluding stop words and 
    invalid tokens. Optionally returns token start and end indices.

Tokenizer._bigrams(list_):
    Generates bigrams from a list of tokens.

Tokenizer._trigrams(list_):
    Generates trigrams from a list of tokens.

Tokenizer.ngrams(list_):
    Combines bigrams and trigrams for a list of tokens.

Tokenizer.ngrams_and_indexes(str_):
    Tokenizes input text, generates n-grams, and returns both n-grams and their corresponding indices.

Tokenizer._ngram_index(indexes):
    Computes the start and end indices for n-grams based on the indices of individual tokens.

Dependencies:
-------------
- string: Used for handling punctuation.
- spacy: NLP library for tokenization and linguistic processing. Requires the 'en_core_web_sm' model.
"""

import string
import spacy

class Tokenizer(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser','ner'])
    def _remove_punctuation(word):
        return ''.join([char for char in word if not char in string.punctuation+' '])
    def _is_valid_word(word):
        return not word.isnumeric() and len(Tokenizer._remove_punctuation(word.replace('\n', '')))!=0 and len(word)>1
    def tokenize(self, str_, include_index=False):
        tokens = [word.lemma_.lower().replace('\n','') for word in self.nlp(str_) if not word.is_stop and word.lemma_.isalnum()]
        # tokens =  # [word.replace('\n', '') for word in tokens if ]
        if not include_index:
            return filter(Tokenizer._is_valid_word, tokens)
        else:
            idx = [(word.idx, word.idx+len(word)) for word in self.nlp(str_) if not word.is_stop and word.lemma_.isalnum()]
            return filter(Tokenizer._is_valid_word, tokens), idx
    
    def _bigrams(list_):
        return list(zip(list_[:-1], list_[1:]))
    def _trigrams(list_):
        return list(zip(list_[:-2], list_[1:-1], list_[2:]))
    def ngrams(list_):
        list_=list(list_)
        return Tokenizer._bigrams(list_)+Tokenizer._trigrams(list_)
    
    def ngrams_and_indexes(self,str_):
        tokens, indices = self.tokenize(str_, include_index=True)
        ngram_list = list(tokens)
        ngram_list += [' '.join(ngram) for ngram in Tokenizer.ngrams(ngram_list)]
        indices += Tokenizer._ngram_index(indices)
        return ngram_list, indices
        
    def _ngram_index(indexes):
        final_indexes=[]
        for elem in Tokenizer.ngrams(indexes):
            start = elem[0][0]
            end = elem[-1][-1]
            final_indexes.append((start,end))
        return final_indexes
    
            
#     def ngrams _and_idx(self,text):
#         tokens = [word.lemma_.lower().replace('\n','') for word in self.nlp(str_) if not word.is_stop and word.lemma_.isalnum()]
#         # tokens =  # [word.replace('\n', '') for word in tokens if ]
#         tokens =  filter(Tokenizer._is_valid_word, tokens)
#         ngram_list = list(token_list)
#         ngram_list += [' '.join(ngram) for ngram in self.ngrams(ngram_list)]