import spacy


def get_spacy_doc(model_name):
    print('loading model...')
    nlp = spacy.load(model_name)
    return nlp
