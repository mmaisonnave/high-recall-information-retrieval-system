import re
import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from utils.io import info, ok

if __name__=="__main__":
    
    input_ = ' '.join(sys.argv)
    
    # OUTPUT #1
    output_file = re.findall('--output-file=([^\ ]*)', input_)
    if len(output_file)==0:
        print('Please indicate output file (python script.py --output-file=../precomputed/cities_and_provincies.txt')
        sys.exit(1)
    output_file = output_file[0]
    
    
    # INPUT #1
    input_file = re.findall('--input-file=([^\ ]*)', input_)
    if len(input_file)==0:
        print('Please indicate output file (python script.py --input-file=../precomputed/cgn_canada_csv_eng.txt')
        sys.exit(1)
    input_file = input_file[0]
    
    print('INPUT_FILE:')
    print(input_file) 
    print()
    df = pd.read_csv(input_file)
    info(f'Number of entries detected from {input_file}: {df.shape[0]:,}')
    
#     mask1 = df["Generic Category"]=="Populated Place"
#     mask2 = df["Generic Term"]=="Territory"
#     mask3 = df["Generic Term"]=="Province"

    valid_terms = [
#                    'Territory',
#                    'Province',
                   'City',
#                    'Town',
#                    'Settlement',
#                    'Village',
                  ]
    
    mask = df["Generic Term"]==valid_terms[0]
    for term in valid_terms[1:]:
        mask = mask|(df["Generic Term"]==term)
        
    filter_df = df[mask]
    info(f'Number of entries after filtering: {filter_df.shape[0]:,}')

    filter_df["Geographical Name"].to_csv(output_file, index=False, header=False)
    ok(f'Entries saved to: {output_file}')
