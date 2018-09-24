import spacy_model
import pos

inputs = "When the new employee is entered into WAP, their CID is created"

nlp = spacy_model.get_model('en_core_web_sm')
doc = nlp(inputs)
pos = pos.get_pos(doc)
print(pos)

