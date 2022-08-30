import sys
sys.path.append('..')

import re
import os
import threading
from utils.tdmstudio import get_title_and_text
from utils.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np

# def _bigrams(list_):
#     return list(zip(list_[:-1], list_[1:]))
# def _trigrams(list_):
#     return list(zip(list_[:-2], list_[1:-1], list_[2:]))
# def ngrams(list_):
#     list_=list(list_)
#     return _bigrams(list_)+_trigrams(list_)

if __name__=='__main__':
    print(f'[  OK   ] Computing freq of n-grams, starting...')
    tokenizer = Tokenizer()
    
    # INPUT #1: output_folder
    input_ = ' '.join(sys.argv)
    output_folder = re.findall('--output-file=([^\ ]*)', input_)
    if len(output_folder)==0:
        print('Please indicate output file (python script.py --output-file=../precomputed/ngram_freq.txt')
        sys.exit(1)
    output_folder = output_folder[0]

    # INPUT #2: unigram freq
    unigram_file = re.findall('--unigram-file=([^\ ]*)', input_)
    if len(unigram_file)==0:
        print('Please indicate output file (python script.py --unigram-file=../precomputed/uni_gram_freq.txt')
        sys.exit(1)
    unigram_file = unigram_file[0]
    
    # INPUT #3: data_sources
    data_sources = [arg for arg in sys.argv[1:] if '--output-file' not in arg]
    data_sources = [data_source for data_source in data_sources if os.path.exists(data_source)]
    if len(data_sources)==0:
        print('Please provide data sources (/home/ec2-user/SageMaker/data/GM_all_1945_1956/)')
        sys.exit(1)
        
    # Reading files from data_sources
    files = [data_source+file for data_source in data_sources for file in os.listdir(data_source)]
        
    # INPUT #4: random sample
    random_sample = re.findall('--random-sample=([^\ ]*)', input_)
    if len(random_sample)>0:
        print('[WARNING] Using random sample only.')
        random_sample = int(random_sample[0])  
        ran = np.random.default_rng(2021)
        files = ran.choice(files, size=random_sample,replace=False)
        
    print(f'[  OK   ] Processing {len(files)} files')
        

    

    
    ## 

    unigram_freq = ([(line.split(';')[0], int(line.split(';')[1])) 
                 for line in open(unigram_file, 'r').read().splitlines()])
    unigram_freq = sorted(unigram_freq, key=lambda x: x[1], reverse=True)[:10000]
    unigram_freq = dict(unigram_freq)
    len(unigram_freq)


    ngram_freq = {}
    count=1
    for token_list in tqdm(map(lambda file: tokenizer.tokenize(get_title_and_text(file)), files)):
        ngram_list =  Tokenizer.ngrams(token_list)
        ngram_list = filter(lambda ngram: all([unigram in unigram_freq for unigram in ngram]) , ngram_list)

        for ngram in set(ngram_list):
            if not ngram in ngram_freq:
                ngram_freq[ngram]=1
            else:
                ngram_freq[ngram]+=1
        if count%10000==0: # 1% (0.01)
            sorted_list = sorted([(word,ngram_freq[word]) for word in ngram_freq], key=lambda x: x[1],reverse=True)[:20000]
            ngram_freq = dict(sorted_list)
            # DEBUGGING CAN BE REMOVED
            sorted_list = sorted([(word,ngram_freq[word]) for word in ngram_freq], key=lambda x: x[1],reverse=True)
            with open(output_folder,  'w') as w:
                w.write('\n'.join([' '.join(word)+f';{count}' for word, count in sorted_list[:20000]]))
            # DEBUGGING CAN BE REMOVED

        count+=1

    sorted_list = sorted([(word,ngram_freq[word]) for word in ngram_freq], key=lambda x: x[1],reverse=True)

#     freq_filename = 'precomputed/ngram_freq.txt'
    with open(output_folder,  'w') as w:
        w.write('\n'.join([' '.join(word)+f';{count}' for word, count in sorted_list[:20000]]))
