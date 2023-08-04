import os 
import re
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('..')

from utils.tdmstudio import get_title_and_text
from utils.tokenizer import Tokenizer
from utils.io import info,warning


if __name__=='__main__':
    input_ =  ' '.join(sys.argv)

    if re.search('--help', input_):
        print('[USAGE]')
        print(f'python {sys.argv[0]} --output-file=..precomputed/idf.txt '\
                '--debug --unigram-file=../precomupted/uni_gram_freq.txt '\
                '--ngram-file=../precomputed/ngram_freq.txt '\
                '--vocab-file=../precomputed/vocab.txt '\
                '--rancom-sample=5000000'\
                '--add=multicultural'
                )
        print()
        sys.exit(0)
    
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
        
    files = [data_source+file for data_source in data_sources for file in os.listdir(data_source)]
    
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
    
    
    # INPUT #7: random sample
    random_sample = re.findall('--random-sample=([^\ ]*)', input_)
    if len(random_sample)>0:
        print('[WARNING] Using random sample only.')
        random_sample = int(random_sample[0])  
        ran = np.random.default_rng(2020)
        files = ran.choice(files, size=random_sample,replace=False)
        
    # INPUT #8: additional words
    additional=re.findall('--add=([^\ ]*)', input_)
    if len(additional)!=len(set(additional)):
        warning(f'Repeated additional words. Removing duplicates {len(additional)-len(set(additional))}...')
        additional = list(set(additional))
        info(f'Aditional words: {additional}')
    
        
    
    tokenizer = Tokenizer()

    freq = [(line.split(';')[0], int(line.split(';')[1])) for line in open(unigram_file).read().splitlines()]
    freq += [(line.split(';')[0], int(line.split(';')[1])) for line in open(ngram_file).read().splitlines()]

    freq = sorted(freq, key=lambda x: x[1],reverse=True)[:10000]

    for word in set(additional).intersection(set([word for word,frecuency in freq])):
        warning(f'Word already in vocabulary, removing: {word}')
        additional.remove(word)

    for idx,word in enumerate(additional):
        info(f'Removing word {freq[-(idx+1)][0]:20} for additional: {word:20} ')

    #freq=dict(freq)
    if len(additional)>0:
        vocab = list([word for word,frequency in freq[:-len(additional)]])
        info(f'Size of vocab without additionals={len(vocab)}')
        vocab = vocab + additional
        for word in additional:
            info(f'Including {word:20} in the vocabulary')

        info(f'Size of vocab with    additionals={len(vocab)}')
    else:
        vocab = list([word for word,frequency in freq])

    
    with open(vocab_file, 'w') as f:
        f.write('\n'.join(vocab))
        

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

