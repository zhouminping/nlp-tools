import models
import dependency_parser_allennlp


dependency_predictor = models.MODELS['dependency-parsing'].predictor()
input = "When the new employee is entered into SAP, their NUID is created and this information is sent to the Ncard System"
dependency_parser_allennlp = dependency_parser_allennlp.DependencyParser(dependency_predictor)
dependency = dependency_parser_allennlp.parse(input)
print(dependency)


# input_sentence = {"sentence": input.strip()}
# json_input = dependency_predictor.load_line(json.dumps(input_sentence))
# print(json_input)
# json_output = dependency_predictor.predict_json(json_input)
# output = dependency_predictor.dump_line(json_output)
# print(output)
#
# semantic_role_labeling_predictor = models.MODELS['semantic-role-labeling'].predictor()
# json_output = semantic_role_labeling_predictor.predict_json(json_input)
# output = dependency_predictor.dump_line(json_output)
# print(output)