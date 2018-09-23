import chunker
import spacy_model

inputs = "When the new employee is entered into SAP, their NUID is created and this information is sent to the Ncard System"

nlp = spacy_model.get_model('en_core_web_sm')
doc = nlp(inputs)
chunk = chunker.get_noun_chunk(doc)
print(chunk)