import spacy
def preprocessor(text, nlp=spacy.load('en_core_web_lg', disable=['textcat', 'parser', 'ner'])):
    return ' '.join([token.lemma_.lower() for token in nlp(text) if token.lemma_.isalpha()])


from sklearn.feature_extraction.text import TfidfVectorizer
def get_default_vectorizer():
    nlp = spacy.load('en_core_web_sm')
    stopwords = {stopword for stopword in nlp.Defaults.stop_words if stopword==preprocessor(stopword)}
    return TfidfVectorizer(lowercase=True,
                           preprocessor=preprocessor,
                           stop_words=stopwords,
                           ngram_range=(1,3),
                           max_features=10000,
                           use_idf=True,
                           smooth_idf=True,
                          )