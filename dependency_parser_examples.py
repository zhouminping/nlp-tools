import allennlp_model
import dependency_parser
import time


inputs = "When the new employee is entered into SAP, their NUID is created and this information is sent to the Ncard System"

start = time.time()
dependency_predictor = allennlp_model.MODELS['dependency-parsing'].predictor()
end = time.time()
print('get dependency_predictor: ', str(end - start))

start = time.time()
dependency_parser = dependency_parser.DependencyParser(dependency_predictor)
end = time.time()
print('get dependency_parser: ', str(end-start))

start = time.time()
dependency = dependency_parser.parse(inputs)
end = time.time()
print('get the parse tree: ', str(end-start))
print(dependency)

start = time.time()
dependency = dependency_parser.one_one_dependency(inputs)
end = time.time()
print('get the one-one dependency: ', str(end-start))
print(dependency)

start = time.time()
dependency = dependency_parser.phrase_dependency(inputs)
end = time.time()
print('get the phrase dependency: ', str(end-start))
print(dependency)
