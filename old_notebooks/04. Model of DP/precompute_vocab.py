from utils.general import info,ok, warning
import os

GM1 = '/home/ec2-user/SageMaker/data/GM_all_1945_1956/'
GM2 = '/home/ec2-user/SageMaker/data/GM_all_1957-1967/'

files = [GM1+f for f in os.listdir(GM1)]+[GM2+f for f in os.listdir(GM2)]
info(f'Cantidad de archivos a procesar: {len(files):,}')

from collections import defaultdict
import spacy
from utils.tdmstudio import TDMStudio
import string
from tqdm import tqdm

# AUXs
info('Loading NLP(spacy) model.')
nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser','ner'])

def remove_punctuation(word):
    return ''.join([char for char in word if not char in string.punctuation+' '])

def tokenize(str_):
    tokens = [word.lemma_.lower() for word in nlp(str_) if not word.is_stop and word.lemma_.isalnum()]
    tokens = [word.replace('\n', '') for word in tokens if not word.isnumeric() and len(remove_punctuation(word.replace('\n', '')))!=0]
    return tokens

info('Creating dictionary')
freq = defaultdict(int)
from threading import Lock
flock = Lock()
wlock = Lock()

import datetime

writer = open('done_vocab.txt', 'w')
writer.write(f'{datetime.datetime.now()} Starting...\n')

# for file_ in tqdm(files):
def process_file(file_):
    wlock.acquire()
    writer.write(file_+'\n')
    writer.flush()
    wlock.release()

    title, text = TDMStudio.get_title_and_text(file_)
    if not title is None and not text is None:
        tokens = set(tokenize(title) + tokenize(text))
        for token in tokens:
            flock.acquire()
            freq[token]+=1
            flock.release()
        for elem in tokens:
            del(elem)
        del(tokens)
    del(title,text)

from utils.general import info, ok
import concurrent.futures

info('Starting...')

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_file, files, chunksize=1000)

for word in [word for word in freq if freq[word]==1]:
    del(freq[word])
writer.write(f'{datetime.datetime.now()} Done!')
writer.close()
ok('Done!')


# assert all([freq[word]>=2 for word in freq])
info(f'len(freq)={len(freq)}')

info('Saving freq.txt')
writer = open('precomputed/freq.txt', 'w')
for word in freq:
    writer.write(f'{word};{freq[word]}\n')
writer.close()

info('Saving vocab.txt')
writer = open('precomputed/vocab.txt','w')
threshold = int(len(files)*(0.01/100.0))
info(f'threshold={threshold}')
vocab = [word for word in freq if freq[word]>threshold]
info(f'len(vocab)={len(vocab)}')
for word in vocab:
    writer.write(f'{word}\n')
writer.close()
