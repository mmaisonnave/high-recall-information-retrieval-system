import sys
sys.path.append('..')

from utils.scal import SCAL
from utils.oracle import Oracle
from utils.data_item import QueryDataItem

import re 

if __name__=='__main__':
    input_ = ' '.join(sys.argv)
    if len(re.findall('.*N=([0-9][0-9]*).*', input_))==0:
        print('Please provide N (python script.py --N=50).')
        sys.exit(1)

    if len(re.findall('.*--target-recall=([0-9\.][0-9\.]*).*', input_))==0:
        print('Please provide target_recall (python script.py --target-recall=0.8).')
        sys.exit(1)
        
    if len(re.findall('.*--seed=([0-9][0-9]*).*', input_))==0:
        print('Please provide seed (python script.py --seed=123).')
        sys.exit(1)

    N = int(re.findall('.*--N=([0-9][0-9]*).*', input_)[0])
    target_recall=float(re.findall('.*--target-recall=([0-9\.][0-9\.]*).*', input_)[0])
    seed=int(re.findall('.*--seed=([0-9\.][0-9\.]*).*', input_)[0])
    session_name=f'simulation_tr_{int(target_recall*100)}_N_{N}_seed_{seed}'
    print(session_name)

    unlabeled = Oracle.get_collection()
    print(f'starting session_name: {session_name}')
    doc = QueryDataItem('dp canada refugees immigration immigrants person persons')
    doc.set_relevant()

    SCAL(session_name, 
         [doc], 
         unlabeled,
         random_sample_size=N,
         simulation=True,
         target_recall=target_recall,
         seed=seed,
        ).run()