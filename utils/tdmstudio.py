
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