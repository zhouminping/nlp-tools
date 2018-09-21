import model
import ner
import dependency_parser

ner_input = """The new employee must claim their My.UNL account before they can claim their email"""

dependency_input = "The new employee must claim their UNL account before they can claim their email"

nlp = model.get_spacy_doc('en_core_web_md')

# ner test
doc = nlp(ner_input)
entities = ner.get_name_entity(doc, [])
print(entities)


# dependency parser test
doc = nlp(dependency_input)
dependency = dependency_parser.get_dependency(doc)
print(dependency)
noun_dependency = dependency_parser.get_noun_dependency(doc)
print(noun_dependency)
