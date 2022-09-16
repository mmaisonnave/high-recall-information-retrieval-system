import sys
sys.path.append('..')

import re
import os
import threading
from utils.tdmstudio import get_title_and_text
from utils.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np


if __name__=='__main__':
    input_ = ' '.join(sys.argv)
    if re.findall('--help', input_):
        print('[USAGE]')
        print(f'python {sys.argv[0]} --output-file=../precomputed/uni_gram_freq.txt '\
                '--random-sample=500000 /home/ec2-user/SageMaker/data/GM_all_1945_1956/ ')
        print()
        sys.exit(0)
    
    print(f'[  OK   ] Computing freq unigrams, starting...')
    tokenizer = Tokenizer()
    
    # INPUT #1: output_folder
    output_folder = re.findall('--output-file=([^\ ]*)', input_)
    if len(output_folder)==0:
        print('Please indicate output file (python script.py --output-file=../precomputed/uni_gram_freq.txt')
        sys.exit(1)
    output_folder = output_folder[0]

    # INPUT #2: data_sources
    data_sources = [arg for arg in sys.argv[1:] if '--output-file' not in arg]
    data_sources = [data_source for data_source in data_sources if os.path.exists(data_source)]
    if len(data_sources)==0:
        print('Please provide data sources (/home/ec2-user/SageMaker/data/GM_all_1945_1956/)')
        sys.exit(1)
        
    # Reading files from data_sources
    files = [data_source+file for data_source in data_sources for file in os.listdir(data_source)]
    
    # INPUT #3: random sample
    random_sample = re.findall('--random-sample=([^\ ]*)', input_)
    if len(random_sample)>0:
        print('[WARNING] Using random sample only.')
        random_sample = int(random_sample[0])  
        ran = np.random.default_rng(2022)
        files = ran.choice(files, size=random_sample,replace=False)

    
    print(f'[  OK   ] Processing {len(files)} files')

    freq = {}
    count=1
    for token_set in tqdm(map(lambda file: set(tokenizer.tokenize(get_title_and_text(file))), files)):
        for token in token_set:
            if not token in freq:
                freq[token]=1
            else:
                freq[token]+=1
        if count%10000==0:
            sorted_list = sorted([(word,freq[word]) for word in freq], key=lambda x: x[1],reverse=True)[:20000]
            freq = dict(sorted_list)
#             freq = dict([(term,freq[term]) for term in list(filter(lambda term: freq[term]>100 ,freq ))])
            # Regular update of freq to use in preliminary works.
            with open(output_folder,  'w') as w:
                w.write('\n'.join([f'{word};{count}' for word, count in sorted_list[:20000]]))
        count+=1

    sorted_list = sorted([(word,freq[word]) for word in freq], key=lambda x: x[1],reverse=True)

#     freq_filename = 'precomputed/uni_gram_freq.txt'
    with open(output_folder,  'w') as w:
        w.write('\n'.join([f'{word};{count}' for word, count in sorted_list[:20000]]))
