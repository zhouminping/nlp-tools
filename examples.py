import model
import ner
import dependency_parser

ner_input = """The new employee must claim their My.UNL account before they can claim their email"""

dependency_input = "I like New York in Autumn"

nlp = model.get_spacy_doc('en_core_web_sm')

# ner test
doc = nlp(ner_input)
entities = ner.get_name_entity(doc, [])
print(entities)


# dependency parser test
doc = nlp(dependency_input)
# for sentence in doc.sents:
#     print(sentence.root)
for token in doc:
    print(token, [tok.text for tok in token.subtree])
dependency = dependency_parser.get_dependency(doc)
print(dependency)
noun_dependency = dependency_parser.get_noun_dependency(doc)
print(noun_dependency)


doc = nlp('I like New York in Autumn.')
sentences = list(doc.sents)
print(sentences[0].root, [child for child in sentences[0].root.children])
