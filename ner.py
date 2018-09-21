import spacy

'''
Support types: 
PERSON: People, including fictional.
NORP: Nationalities or religious or political groups.
FAC: Buildings, airports, highways, bridges, etc.
ORG: Companies, agencies, institutions, etc.
GPE: Countries, cities, states.
LOC: Non-GPE locations, mountain ranges, bodies of water.
PRODUCT: Objects, vehicles, foods, etc. (Not services.)
EVENT: Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART: Titles of books, songs, etc.
LAW: Named documents made into laws.
LANGUAGE: Any named language.
DATE: Absolute or relative dates or periods.
TIME: Times smaller than a day.
PERCENT: Percentage, including "%".
MONEY: Monetary values, including unit.
QUANTITY: Measurements, as of weight or distance.
ORDINAL: "first", "second", etc.
CARDINAL: Numerals that do not fall under another type.
'''


def get_name_entity(doc, entity_types):
    entities = []
    for ent in doc.ents:
        if not entity_types:
            entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
        elif ent.label_ in entity_types:
            entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
    return entities
