from IPython.core.display import display, HTML
import datetime


def info(str_):
    print(f'{datetime.datetime.now()} [ \033[1;94mINFO\x1b[0m  ] {str_}')
def ok(str_):
    print(f'{datetime.datetime.now()} [  \033[1;92mOK\x1b[0m   ] {str_}')
def warning(str_):
    print(f'{datetime.datetime.now()} [\x1b[1;31mWARNING\x1b[0m] {str_}')
def html(str_=''):
    display(HTML(str_))
    
GM_all_part1 = '/home/ec2-user/SageMaker/data/GM_all_1945_1956/'
GM_all_part2 = '/home/ec2-user/SageMaker/data/GM_all_1957-1967/'
GM_dp_dirpath = '/home/ec2-user/SageMaker/data/GM_DP_and_Canada1945_1967/'
import os
def id2file(id_):
    if type(id_)==int:
        id_=str(int)
    file_1 = GM_all_part1+id_+'.xml'
    file_2 = GM_all_part2+id_+'.xml'
    file_3 = GM_dp_dirpath+id_+'.xml'
    
    if os.path.isfile(file_1):
        return file_1
    elif os.path.isfile(file_2):
        return file_2
    elif os.path.isfile(file_3):
        return file_3
    else:
        return None
        