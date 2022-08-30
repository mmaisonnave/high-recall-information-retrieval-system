import os 
import re
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('..')

from utils.tdmstudio import get_title_and_text
from utils.tokenizer import Tokenizer


if __name__=='__main__':
    input_ =  ' '.join(sys.argv)
    
    # INPUT #1
    output_file = re.findall('--output-file=([^\ ]*)', input_)
    if len(output_file)==0:
        print('Please indicate output file (python script.py --output-file=../precomputed/idf.txt')
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
    unigram_file = re.findall('--unigram-file=([^\ ]*)', input_)
    if len(unigram_file)==0:
        print('Please indicate input file (python script.py --unigram-file=../precomputed/uni_gram_freq.txt')
        sys.exit(1)
    unigram_file = unigram_file[0]
    
    # INPUT #5
    ngram_file = re.findall('--ngram-file=([^\ ]*)', input_)
    if len(ngram_file)==0:
        print('Please indicate input file (python script.py --ngram-file=../precomputed/ngram_freq.txt')
        sys.exit(1)
    ngram_file = ngram_file[0]
    
    # INPUT #6
    vocab_file = re.findall('--vocab-file=([^\ ]*)', input_)
    if len(vocab_file)==0:
        print('Please indicate output file (python script.py --vocab-file=../precomputed/vocab.txt')
        sys.exit(1)
    vocab_file = vocab_file[0]
    
    tokenizer = Tokenizer()

    freq = [(line.split(';')[0], int(line.split(';')[1])) for line in open(unigram_file).read().splitlines()]
    freq += [(line.split(';')[0], int(line.split(';')[1])) for line in open(ngram_file).read().splitlines()]

    freq = sorted(freq, key=lambda x: x[1],reverse=True)[:10000]

    freq=dict(freq)
    vocab = list([word for word in freq])
    
    with open(vocab_file, 'w') as f:
        f.write('\n'.join(vocab))
        
    files = [data_source+file for data_source in data_sources for file in os.listdir(data_source)]
    if debug:
        print('[WARNING] debug activated.')
        files = files[:1]
        
    freq = dict([(word,0) for word in vocab])
    for token_list in tqdm(map(lambda file: tokenizer.tokenize(get_title_and_text(file)), files)):
        ngram_list = list(token_list)
        ngram_list += [' '.join(ngram) for ngram in Tokenizer.ngrams(ngram_list)]

        ngram_list = filter(lambda ngram: ngram in freq , ngram_list)

        for ngram in set(ngram_list):
            freq[ngram]+=1

    N=len(files)

    idf = np.log(N/(1+np.array([freq[word] for word in vocab])))

    with open(output_file,'w') as f:
        f.write('\n'.join([str(value) for value in idf]))

