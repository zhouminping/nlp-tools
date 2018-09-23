def get_pos(doc):
    pos = []
    for token in doc:
        pos.append((token.text, token.pos_))
    return pos
