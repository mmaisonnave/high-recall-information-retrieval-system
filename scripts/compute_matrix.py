import sys
sys.path.append('..')

from utils.io import ok, warning
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import os
from tqdm import tqdm
from utils.tdmstudio import get_title_and_text
from utils.tokenizer import Tokenizer
import re
import pickle

import concurrent.futures

from threading import Lock
wlock = Lock()

if __name__=='__main__':
    input_ = ' '.join(sys.argv[1:])


    if re.search('--help', input_):
        print('[USAGE]')
        print(f'python {sys.argv[0]} /home/ec2-user/SageMaker/data/GM_all_1945_1956/ --debug '\
                '--continue --vocab-file=../precomputed/vocab.txt '\
                '--idf-file=../precomputed/idf.txt '\
                '--output-folder=../data/precomputed/'
             )
        print()
        sys.exit(0)


    ok('Starting computing tf-idf vectors ...')
    tokenizer = Tokenizer()

    # input_ =  ' --output-file=precomputed/tfidf_matrix.npz /home/ec2-user/SageMaker/data/GM_all_1945_1956/ '\
    # ' --vocab-file=precomputed/vocab.txt --idf-file=precomputed/idf.txt --debug' 
    input_ = ' '.join(sys.argv[1:])
    ok('Analyzing input ...')
    # INPUT #1
#     output_file = re.findall('--output-file=([^\ ]*)', input_)
#     if len(output_file)==0:
#         print('Please indicate output file (python script.py --output-file=../precomputed/tfidf_matrix.npz')
#         sys.exit(1)
#     output_file = output_file[0]

    # INPUT #2
    data_sources = [arg for arg in sys.argv[1:] if '--output-file' not in arg]
    data_sources = [data_source for data_source in data_sources if os.path.exists(data_source)]
    if len(data_sources)==0:
        print('Please provide data sources (/home/ec2-user/SageMaker/data/GM_all_1945_1956/)')
        sys.exit(1)

    # INPUT #3 
    debug = False
    debug = len(re.findall('--debug', input_))>0

    # INPUT #3.1
    continue_ = False
    continue_ = len(re.findall('--continue', input_))>0

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
    output_folder=re.findall('--output-folder=([^\ ]*)', input_)
    if len(output_folder)==0:
        print('Please indicate input file (python script.py --output-folder=../data/precomputed/')
        sys.exit(1)
    output_folder=output_folder[0]
    
    ###        
    vocab = open(vocab_file, 'r').read().splitlines()
    word2index = dict([(word,idx) for idx,word in enumerate(vocab)])


    files = [data_source+file for data_source in data_sources for file in os.listdir(data_source)]
    if debug:
        warning('debug activated.')
        files = files[:10]
        
    procced_file='computing_tfidf_all_corpus.txt'

    if continue_:
        processed = set([line for line in open(procced_file,'r').read().splitlines()])
        count_before=len(files)
        files = list(filter(lambda x: x.split('/')[-1][:-4] not in processed, files))
        warning(f'Continuing from file no. {len(processed)}   (processing {len(files)} files instead of {count_before})')
        
    ok(f'Vocab size={len(vocab)}, Number of files={len(files)}')
    # with open(column_file, 'w') as f:
    #     f.write('\n'.join(map(lambda x: x.split('/')[-1], files)))

#     m = sparse.lil_matrix((len(files), len(vocab)))
    idf = sparse.lil_matrix(np.array([float(value) for value in open(idf_file, 'r').read().splitlines()]).reshape(1,-1))

    
    ok(f'Saving output in:  {procced_file}')
    writer = open(procced_file,'a')

    def process_file(file):
        id_ = file.split('/')[-1][:-4]
        precomputed_file = os.path.join(output_folder,file.split('/')[-1][:-4])
        precomputed_file = precomputed_file+'_representations.p'
        if os.path.isfile(precomputed_file):
            vectors = pickle.load(open(precomputed_file,'rb'))
        else:
            vectors={}
        # Empty representation        
        vec = sparse.lil_matrix((1, len(vocab)))
        
        # Tri-, bi-, and unigrams in file
        token_list = tokenizer.tokenize(get_title_and_text(file))
        ngram_list = list(token_list)
        ngram_list += [' '.join(ngram) for ngram in Tokenizer.ngrams(ngram_list)]
        ngram_list = filter(lambda ngram: ngram in word2index , ngram_list)
        
        # Update representation
        for ngram in ngram_list:
            vec[0, word2index[ngram]]+=1
        # Apply IDF
        vec = vec.multiply(idf)
        # Normalize
        vec = normalize(vec, axis=1)
        vectors['BoW']=vec
        # Save representation
        pickle.dump(vectors, open(precomputed_file, 'wb'))

        wlock.acquire()
        writer.write(f'{id_}\n')
        writer.flush()
        wlock.release()
    ok('Starting concurrent process')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_file, files, chunksize=4000)
    writer.close()
    ok('FINISHED!')
