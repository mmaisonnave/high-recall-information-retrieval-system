"""
This module provides utilities for processing data files in the format provided by TDM Studio datasets. 

It includes functions to:
- Extract and clean the title and text content from TDM XML files.
- Retrieve the title only from TDM XML files.
- Locate a file in a specified root directory by its ID.
- Extract the publication date from TDM XML files.

Dependencies:
- `lxml` for XML parsing.
- `bs4` (BeautifulSoup) for cleaning extracted text.

Functions:
- `get_title_and_text(filename: str) -> str`: Extracts and concatenates the title and body text of a TDM XML file.
- `get_title(filename: str) -> str`: Extracts the title of a TDM XML file.
- `get_filename(id_: str) -> str`: Finds the full path of an XML file based on a given ID.
- `get_date(filename: str) -> str`: Extracts the publication date from a TDM XML file.

"""
from lxml import etree
from bs4 import BeautifulSoup

def get_title_and_text(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    if root.find('.//HiddenText') is not None:
        text = (root.find('.//HiddenText').text)

    elif root.find('.//Text') is not None:
        text = (root.find('.//Text').text)

    else:
        text = None
    title = root.find('.//Title').text
    str_=''
    if not title is None:
        str_+=f'{title}.'
    if not text is None:
        str_+=f'{text}.'

    return BeautifulSoup(str_, parser='html.parser', features="lxml").get_text()


def get_title(filename:str) -> str:
    tree = etree.parse(filename)
    root = tree.getroot()
   
    title = root.find('.//Title').text
    str_=''
    if not title is None:
        str_+=f'{title}.'

    return BeautifulSoup(str_, parser='html.parser', features="lxml").get_text()

import os
def get_filename(id_):
    root ='/home/ec2-user/SageMaker/data'
    data_sources = [os.path.join(root, folder) for folder in os.listdir(root)]
    for data_source in data_sources:
        if os.path.isfile(os.path.join(data_source,id_+'.xml')):
            return os.path.join(data_source,id_+'.xml')
        
        
def get_date(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    date = root.find('.//NumericDate').text
    return date
