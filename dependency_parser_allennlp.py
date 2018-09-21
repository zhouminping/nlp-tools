from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
import utils


class DependencyParser():

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def parse(self, inputs: str) -> {}:
        json_dict_str = utils.to_json_dict_str(inputs)
        json_input = self.predictor.load_line(json_dict_str)
        output = self.predictor.predict_json(json_input)
        hierplane_tree = output['hierplane_tree']
        text = hierplane_tree["text"]
        parse_tree = [("root", hierplane_tree['root']['word'])]
        children = hierplane_tree['root']['children']
        for child in children:
            node_type = child['nodeType']
            if node_type not in ['punct', 'aux', 'auxpass']:
                spans = {'start': [], "end": []}
                self.__traverse__(child, spans)
                parse_tree.append((node_type, self.__get_phrase__(text, spans)))
        return parse_tree

    def __traverse__(self, node: JsonDict, spans: {}) -> None:
        start = node['spans'][0]['start']
        end = node['spans'][0]['end']
        spans['start'].append(start)
        spans['end'].append(end)
        if 'children' not in node:
            return
        for child in node['children']:
            self.__traverse__(child, spans)

    @staticmethod
    def __get_phrase__(text: str, spans: {}) -> str:
        min_start = min(spans['start'])
        max_end = max(spans['end'])
        return text[min_start: max_end]
