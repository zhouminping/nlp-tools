from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict


class DependencyParser:

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def one_one_dependency(self, inputs: str) -> []:
        dependency_tree = self.parse(inputs)
        root = dependency_tree['root']
        dep = []
        self.__get_one_one_dependency(root, dep)
        return dep

    def __get_one_one_dependency(self, node: JsonDict, dep: []) -> None:
        if 'children' not in node:
            return
        for child in node['children']:
            dep.append((child['word'], child['spans'][0]['start'], child['spans'][0]['end']-2,
                        node['word'], node['spans'][0]['start'], node['spans'][0]['end']-2, child['nodeType']))
            self.__get_one_one_dependency(child, dep)

    def phrase_dependency(self, inputs: str) -> []:
        dependency_tree = self.parse(inputs)
        text = dependency_tree["text"]
        parse_tree = [("root", (dependency_tree['root']['word'],
                                dependency_tree['root']['spans'][0]['start'],
                                dependency_tree['root']['spans'][0]['end']-2))]
        children = dependency_tree['root']['children']
        for child in children:
            node_type = child['nodeType']
            spans = {'start': [], "end": []}
            self.__get_phrase_span(child, spans)
            parse_tree.append((node_type, self.__get_phrase__(text, spans)))
        return parse_tree

    def parse(self, inputs: str) -> JsonDict:
        output = self.predictor.predict(inputs)
        return output['hierplane_tree']

    def __get_phrase_span(self, node: JsonDict, spans: {}) -> None:
        start = node['spans'][0]['start']
        end = node['spans'][0]['end']
        spans['start'].append(start)
        spans['end'].append(end)
        if 'children' not in node:
            return
        for child in node['children']:
            self.__get_phrase_span(child, spans)

    @staticmethod
    def __get_phrase__(text: str, spans: {}) -> ():
        min_start = min(spans['start'])
        max_end = max(spans['end'])
        return text[min_start: max_end-1], min_start, max_end-2
