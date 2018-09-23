import spacy


def get_model(model_name):
    print('loading spacy model...')
    nlp = spacy.load(model_name)
    return nlp
