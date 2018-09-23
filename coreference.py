from allennlp.predictors import Predictor

class Coreference:

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def get_coreference(self, inputs: str) -> []:
        information = self.predictor.predict(inputs)
        words = information['document']
        clusters = information['clusters']
        corefs = []
        for cluster in clusters:
            # print(cluster)
            coref = []
            for ele in cluster:
                coref.append((words[ele[0]:ele[1]+1], ele[0], ele[1]))
            corefs.append(coref)
        return corefs
