import sys
sys.path.append('..')

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import os
from tqdm import tqdm
from utils.tdmstudio import get_title_and_text
from utils.tokenizer import Tokenizer
import re


if __name__=='__main__':
    tokenizer = Tokenizer()
    
    input_ =  ' '.join(sys.argv)
    
    # INPUT #1
    output_file = re.findall('--output-file=([^\ ]*)', input_)
    if len(output_file)==0:
        print('Please indicate output file (python script.py --output-file=../precomputed/BoW_matrix.npz')
        sys.exit(1)
    output_file = output_file[0]
    
    # INPUT #2
    data_sources = [arg for arg in sys.argv[1:] if '--output-file' not in arg]
    data_sources = [data_source for data_source in data_sources if os.path.exists(data_source)]
    if len(data_sources)==0:
        print('Please provide data sources (/home/ec2-user/SageMaker/data/GM_all_1945_1956/)')
        sys.exit(1)
        
    # INPUT #3 
    debug = False
    debug = len(re.findall('--debug', input_))>0
   
    
    # INPUT #4
    vocab_file = re.findall('--vocab-file=([^\ ]*)', input_)
    if len(vocab_file)==0:
        print('Please indicate input file (python script.py --vocab-file=../precomputed/vocab.txt')
        sys.exit(1)
    vocab_file = vocab_file[0]
    
    # INPUT #5
    idf_file = re.findall('--idf-file=([^\ ]*)', input_)
    if len(idf_file)==0:
        print('Please indicate input file (python script.py --idf-file=../precomputed/idf.txt')
        sys.exit(1)
    idf_file = idf_file[0]
    
    # INPUT #6
    column_file = re.findall('--columns-file=([^\ ]*)', input_)
    if len(column_file)==0:
        print('Please indicate output file (python script.py --columns-file=../precomputed/columns.txt')
        sys.exit(1)
    column_file = column_file[0]
    
    
    ###        
    vocab = open(vocab_file, 'r').read().splitlines()
    word2index = dict([(word,idx) for idx,word in enumerate(vocab)])

    
    files = [data_source+file for data_source in data_sources for file in os.listdir(data_source)]
    if debug:
        print('[WARNING] debug activated.')
        files = files[:10000]

    with open(column_file, 'w') as f:
        f.write('\n'.join(map(lambda x: x.split('/')[-1], files)))

    m = sparse.lil_matrix((len(files), len(vocab)))


    count = 0
    for token_list in tqdm(map(lambda file: tokenizer.tokenize(get_title_and_text(file)), files)):
        ngram_list = list(token_list)
        ngram_list += [' '.join(ngram) for ngram in Tokenizer.ngrams(ngram_list)]

        ngram_list = filter(lambda ngram: ngram in word2index , ngram_list)
        for ngram in ngram_list:
            m[count, word2index[ngram]]+=1
        count+=1

    idf = sparse.lil_matrix(np.array([float(value) for value in open(idf_file, 'r').read().splitlines()]).reshape(1,-1))

    
    m = m.multiply(idf)    
    m = normalize(m, axis=1)

    sparse.save_npz(output_file, m)