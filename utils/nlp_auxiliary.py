import spacy
def preprocessor(text, nlp=spacy.load('en_core_web_lg', disable=['textcat', 'parser', 'ner'])):
    return ' '.join([token.lemma_.lower() for token in nlp(text) if token.lemma_.isalpha()])