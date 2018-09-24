import spacy_model
import ner

inputs = """Book me a Mars room at 3:00 afternoon on Sept. 27th"""

nlp = spacy_model.get_model('en_core_web_md')

# ner test
doc = nlp(inputs)
entities = ner.get_name_entity(doc)
print(entities)
