import os
import pickle
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.io import ok,info,warning

from threading import Lock
wlock = Lock()

import concurrent.futures

if __name__=='__main__':
    precomputed_folder='/home/ec2-user/SageMaker/mariano/notebooks/04. Model of DP/precomputed'
    files = [file for file in os.listdir(precomputed_folder) if file.endswith('glove.p')]
    info(f'Processing {len(files):,} files.')
    
    temporary_output_file = '../precomputed/updated_precomputed_vectors.txt'
    
    if os.path.isfile(temporary_output_file):
        already_processed = (open(temporary_output_file,'r').read().splitlines())
        
        processed_with_errors = [item[:-len(',ERROR')] for item in already_processed if 'ERROR' in item]
        correctly_processed = [item[:-len(',OK')] for item in already_processed if 'OK' in item]
        
        already_processed=set(processed_with_errors+correctly_processed)
        
        files = [file for file in files if not file.split('.')[0] in already_processed]        
        warning(f'Already processed files found ({len(processed_with_errors):,} files with errors '\
                f'and {len(correctly_processed)} correctly processed ).')
        warning(f'Now processing {len(files):,} files.')
    else:
        info(f'Creating file to store current progress ({temporary_output_file})')
    
    writer = open(temporary_output_file, 'a',)
    def process_file(file):
        id_ = file.split('.')[0]
        wlock.acquire()
        writer.write(f'{id_},')
        writer.flush()
        wlock.release()
        try:
            list_ = pickle.load(open(os.path.join(precomputed_folder,file), 'rb'))

            assert len(list_)==5

            assert list_[1].shape==(600,)
            del(list_[1])

            assert list_[0].shape==(300,)

            assert list_[1].shape==(1,10000)
            del(list_[1])

            assert list_[1].shape==(1,10000)
            del(list_[1])

            assert list_[0].shape==(300,)
            assert list_[1].shape==(1,10000)
            assert len(list_)==2

            pickle.dump(list_, open(os.path.join(precomputed_folder,file), 'wb'))
            wlock.acquire()
            writer.write(f'OK\n')
            writer.flush()
            wlock.release()
        except:
            wlock.acquire()
            writer.write(f'ERROR\n')
            writer.flush()
            wlock.release()
            
    info('Starting concurrent process')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_file, files, chunksize=4000)
    writer.close()
    ok('FINISHED!')
            
    writer.close()        