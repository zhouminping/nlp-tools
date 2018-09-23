def get_noun_chunk(doc):
    chunks = []
    for chunk in doc.noun_chunks:
        chunks.append((chunk.text, chunk.start_char, chunk.end_char))
    return chunks
