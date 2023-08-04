import re
import sys
sys.path.append('..')
from utils.io import ok, info, ask_confirmation, warning

import os
from utils.tdmstudio import get_title_and_text

if __name__=='__main__':
    input_ = ' '.join(sys.argv)
    
    remove = False
    if len(re.findall('--remove-files', input_))!=0:
        warning('Removing source folders, cannot be undone.')
        remove = ask_confirmation()
    else:
        info('Remember using --remove--files to update the source folder.')
    
    info('' if remove else 'NOT '+'Removing files from source folder.')
    
    # INPUT #2
    data_sources = [arg for arg in sys.argv[1:] if '--output-file' not in arg]
    data_sources = [data_source for data_source in data_sources if os.path.exists(data_source)]
    if len(data_sources)==0:
        print('Please provide data sources (/home/ec2-user/SageMaker/data/GM_all_1945_1956/)')
        sys.exit(1)        
    for idx, data_source in enumerate(data_sources):
        info(f'Using data source #{idx+1}: {data_source}')
        
    ########
    # CODE #
    ########
#     keywords = [line for line in open(input_file).read().splitlines()]
#     keywords += ['Alberta', 'AB',
    keywords = ['Alberta', '[(\ ]AB', 'Albertan',
                'British Columbia', '[(\ ]BC', 'British Columbian',
                'Manitoba',  '[(\ ]MB', 'Manitoban',
                'New Brunswick',  '[(\ ]NB', 'New Brunswicker'
                'Newfoundland and Labrador', '[(\ ]NL', 'Newfoundlander Labradorian'
                'Nova Scotia',  '[(\ ]NS', 'Nova Scotian'
                'Ontario', '[(\ ]ON', 'Ontarian',
                'Prince Edward Island', '[(\ ]PE', 'Prince Edward Islander' 
                'Quebec',  '[(\ ]QC', 'Quebecer', 'Quebecker',
                'Saskatchewan', '[(\ ]SK', 'Saskatchewanian',
                'Northwest Territories', '[(\ ]NT', 'Northwest Territorian',
                'Nunavut', '[(\ ]NU','Nunavummiut', 'Nunavummiuq'
                'Yukon', '[(\ ]YT', 'Yukoners',
                'Canada', '[(\ ]CA', 'Canadian' ]
    

    info(f'Using {len(keywords)} entities (country, provincies, territories and demonyms).')
    
    regex = re.compile('|'.join([keyword for keyword in keywords]))    

        
    files = [data_source+file for data_source in data_sources for file in os.listdir(data_source)]
    info(f'Processing {len(files)} files.')    
    
#     # BEGIN DEBUG #
#     info('BEGIN DEBUG')
#     regex1 = re.compile('[(]NS') 
# #     regex2 = re.compile('NS')
#     count=0
#     for file in files:
#         match_ = regex1.search(get_title_and_text(file))
#         if match_:
#             print(f'MATCH: {match_.string[match_.start()-10:match_.end()+10]}')
#             count+=1
#         if count==100:
#             break
#     info('END   DEBUG')
#     # END   DEBUG #
    
    def not_about_canada(file):
        return regex.search(get_title_and_text(file)) is None   
           
    def process_file(file):
        if not_about_canada(file):
            os.remove(file)
            return True
        return False
    
    if remove:
        deleted = list(map(process_file,files))
        warning(f'Removed: {len([elem for elem in deleted if elem==True])} files.')
    else:
        to_remove = list(filter(not_about_canada, files))
        info(f'No of files found to remove: {len(to_remove)}')

    ok(f'FINISH')
