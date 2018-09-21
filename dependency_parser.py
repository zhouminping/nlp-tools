

def get_noun_dependency(doc):
    dependency = []
    for chunk in doc.noun_chunks:
        dependency.append((chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text))
    return dependency


def get_dependency(doc):
    dependency = []
    for token in doc:
        dependency.append((token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children]))
    return dependency


