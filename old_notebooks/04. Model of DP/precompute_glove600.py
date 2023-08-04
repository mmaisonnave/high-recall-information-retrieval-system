import os
import numpy as np

GM1 = '/home/ec2-user/SageMaker/data/GM_all_1945_1956/'
GM2 = '/home/ec2-user/SageMaker/data/GM_all_1957-1967/'

files = [GM1+f for f in os.listdir(GM1)]+[GM2+f for f in os.listdir(GM2)]
print(f'len(files)={len(files)}')

import spacy
from numpy import load,save
from utils.tdmstudio import TDMStudio
nlp = spacy.load('en_core_web_lg', disable=['textcat','ner','parser','lemmatizer','tagger'])
nlp.tokenizer

def process_file(file_):
    output = f"./precomputed/{file_.split('/')[-1][:-4]}_glove600.npy"
    pickle_file = output.replace('_glove600.npy', '_glove.p')
    if not os.path.isfile(output) and not os.path.isfile(pickle_file): ###################################### AND IF NOT EXIST GLOVE.p
        title,text = TDMStudio.get_title_and_text(file_)
        if not title is None and not text is None:
            X = np.zeros(shape=(600,))
            X[:300] = nlp.tokenizer(title).vector
            X[300:] = nlp.tokenizer(text).vector
            save(output, X)
            del(X)
        del(title,text)
    del(output)

from utils.general import info, ok
import concurrent.futures
import datetime
writer = open('precompute_glove600.out', 'w')
writer.write(f'{datetime.datetime.now()} Starting...\n')

info('Starting...')

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_file, files)


writer.write(f'{datetime.datetime.now()} Done!\n')
writer.close()
ok('Done!')
