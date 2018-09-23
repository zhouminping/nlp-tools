from allennlp.predictors import Predictor


class SemanticRoleLabeling:

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def parse(self, inputs: str) -> {}:
        parsing = self.predictor.predict(inputs)
        semantic_role = {}
        words = parsing['words']
        verbs = parsing['verbs']
        for semantic in verbs:
            verb = semantic['verb']
            tags = semantic['tags']
            role = self.__get_semantic_role(words, tags)
            if role:
                if verb not in semantic_role:
                    semantic_role[verb] = []
                semantic_role[verb] = role
        return semantic_role

    @staticmethod
    def __get_semantic_role(words, tags):
        semantic_role = []
        index = 0
        while index < len(tags):
            if tags[index].startswith('B'):
                role = tags[index][2:]
                if role != 'V':
                    phrase = ""
                    while index < len(tags) and tags[index] != 'O':
                        phrase += words[index] + " "
                        index += 1
                    semantic_role.append((phrase.strip(), role))
            index += 1
        return semantic_role


