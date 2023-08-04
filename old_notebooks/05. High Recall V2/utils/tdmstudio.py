import pandas as pd
from lxml import etree
from bs4 import BeautifulSoup
import re

class TDMStudio(object):

    def get_date(filename):
        tree = etree.parse(filename)
        root = tree.getroot()
    #     title = root.find('.//Title').text
        date = root.find('.//NumericDate').text
    #     publisher = root.find('.//PublisherName').text
        assert date is not None

        return date

    def get_title_and_text(filename):
        tree = etree.parse(filename)
        root = tree.getroot()
        if root.find('.//HiddenText') is not None:
            text = (root.find('.//HiddenText').text)

        elif root.find('.//Text') is not None:
            text = (root.find('.//Text').text)

        else:
            text = None

        title = root.find('.//Title')
        if title is not None:
            title = title.text
        if not text is None:
            text = BeautifulSoup(text, features='lxml').get_text()
        return title,text
    
    def get_paragraphs(filename):
        tree = etree.parse(filename)
        root = tree.getroot()
        if root.find('.//HiddenText') is not None:
            text = (root.find('.//HiddenText').text)

        elif root.find('.//Text') is not None:
            text = (root.find('.//Text').text)

        else:
            text = None
        paragraphs = []
        if not text is None:
            paragraphs = re.findall('<p>[^<]*</p>',text)
        return [BeautifulSoup(paragraph, features='lxml').get_text().strip() for paragraph in paragraphs]
    
    def get_title(filename):
        tree = etree.parse(filename)
        root = tree.getroot()

        title = root.find('.//Title')
        if title is not None:
            title = title.text

        return title    