import customized_ner
import ner
import spacy_model
from pathlib import Path


init_model = 'en_core_web_md'

label = 'ANIMAL'
labels = [label]

training_data = [
    ("Horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, label)]
    }),

    ("Do they bite?", {
        'entities': []
    }),

    ("horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, label)]
    }),

    ("horses pretend to care about your feelings", {
        'entities': [(0, 6, label)]
    }),

    ("they pretend to care about your feelings, those horses", {
        'entities': [(48, 54, label)]
    }),

    ("horses?", {
        'entities': [(0, 6, label)]
    })
]

output_dir = Path('models/animal_model')

customized_ner.train(init_model, 'animal_model', training_data, labels, output_dir, 100)


test = "dogs care about people's feelings"
nlp = spacy_model.get_model('models/animal_model')
doc = nlp(test)
entities = ner.get_name_entity(doc)
print(entities)
