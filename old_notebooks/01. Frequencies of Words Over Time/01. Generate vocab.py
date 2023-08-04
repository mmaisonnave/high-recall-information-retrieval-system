
# --------------------------------------------------------------------------------
# Retrieving XML file names
# --------------------------------------------------------------------------------
import os
data_path = "/home/ec2-user/SageMaker/data/refugee_dataset_v1/"
assert os.path.exists(data_path)

files = os.listdir(data_path)
len(files)


# --------------------------------------------------------------------------------
# Auxiliary function (getxmlcontent) for reading the text out of the XML file
# --------------------------------------------------------------------------------
from lxml import etree
from bs4 import BeautifulSoup
import spacy

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner','textcat','lemmatizer'])

# We define a function to get the text content that we need from the XML articles available in our dataset
def getxmlcontent(root):
    if root.find('.//HiddenText') is not None:
        return(root.find('.//HiddenText').text)
    
    elif root.find('.//Text') is not None:
        return(root.find('.//Text').text)
    
    else:
        return None

# --------------------------------------------------------------------------------
# Computing preliminary vocab (unigrams with that appear at least once)
# --------------------------------------------------------------------------------
from collections import defaultdict
total_count = defaultdict(int)
count_per_article = defaultdict(int)
for file in files:
    tree = etree.parse(data_path + file)
    root = tree.getroot()
    
    if getxmlcontent(root) is not None:
        soup = BeautifulSoup(getxmlcontent(root), parser='html.parser', features="lxml")
        text = soup.get_text()
        tokens = [token.text for token in nlp(text) if not token.is_stop and token.is_alpha]
        for token in tokens:
            total_count[token]+=1
        for token in set(tokens):
            count_per_article[token]+=1

assert len(count_per_article)==len(total_count)
# print(f'Vocab size after removing terms that appear less than i articles')

print(' i,    appears more than i times, appears more than in i articles.')
for i in range(41):
    print(f'{i:2}:   '\
          f'{len([word for word in total_count if total_count[word]>i]):8}, '\
          f'{len([word for word in count_per_article if count_per_article[word]>i]):8}')
    
print(f'Removing terms that only appear one time in the corpus '\
      f'(-{len([word for word in total_count if total_count[word]==1])})')
for word in set(total_count):
    if total_count[word]==1:
            del(total_count[word])
            del(count_per_article[word])
print(f'vocab size: {len(total_count)}')
assert len(count_per_article)==len(total_count)

vocab = {}
for word in total_count:
    vocab[word] = {}
    vocab[word]['total_count'] = total_count[word]
    vocab[word]['article_count'] = count_per_article[word]

import pickle
vocab_file = '/home/ec2-user/SageMaker/mariano/notebooks/01. Frequencies of Words Over Time/data/vocab.p'
pickle.dump(vocab, open(vocab_file, 'wb'))
