import allennlp_model
import semantic_role_labeling
import time


inputs = "When the new employee is entered into SAP, their NUID is created"

start = time.time()
semantic_predictor = allennlp_model.MODELS['semantic-role-labeling'].predictor()
end = time.time()
print("get semantic predictor: ", str(end-start))

start = time.time()
semantic_parser = semantic_role_labeling.SemanticRoleLabeling(semantic_predictor)
end = time.time()
print("get semantic parser: ", str(end-start))

start = time.time()
semantic_role = semantic_parser.parse(inputs)
end = time.time()
print("semantic parsing for input: ", str(end-start))
print(semantic_role)



