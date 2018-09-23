# coding: utf8
"""Example of training an additional entity type
This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.
The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.
After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import random
from pathlib import Path
import spacy


# new entity labels
# LABELS = ['ANIMAL']

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
"""
TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("Do they bite?", {
        'entities': []
    }),

    ("horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("horses pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("they pretend to care about your feelings, those horses", {
        'entities': [(48, 54, 'ANIMAL')]
    }),

    ("horses?", {
        'entities': [(0, 6, 'ANIMAL')]
    })
]
"""


def train(init_model: str, new_model_name: str, training_data: [], labels: [], output_dir: Path, n_iter: int) -> None:
    """Set up the pipeline and entity recognizer, and train the new entity."""
    nlp = spacy.load(init_model)  # load existing spaCy model
    ner = nlp.get_pipe('ner')
    for label in labels:
        ner.add_label(label)   # add new entity label to entity recognizer

    optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            for text, annotations in training_data:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35, losses=losses)
            print(losses)

    # save model to output directory
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    nlp.meta['name'] = new_model_name  # rename model
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
