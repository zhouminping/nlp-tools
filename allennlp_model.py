from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive


class Model:

    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)


MODELS = {
    'semantic-role-labeling': Model(
        'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
        'semantic-role-labeling'),
    'dependency-parsing': Model(
        'https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz',
        'biaffine-dependency-parser'),
    'coreference-resolution': Model(
        'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
        'coreference-resolution')
}
