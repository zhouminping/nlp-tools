import allennlp_model
import coreference
import time

inputs = "\"I voted for Nader because he was most aligned with my values,\" She said"

# inputs = """Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye
# Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years
# younger, with whom he shared an enthusiasm for computers. Paul and Bill used a teletype terminal at their high
# school, Lakeside, to develop their programming skills on several time-sharing computer systems. """

start = time.time()
coreference_predictor = allennlp_model.MODELS['coreference-resolution'].predictor()
end = time.time()
print("get coreference_predictor: ", str(end-start))

start = time.time()
corefer = coreference.Coreference(coreference_predictor)
end = time.time()
print("get coreference obj: ", str(end-start))

start = time.time()
coref = corefer.get_coreference(inputs)
end = time.time()
print("get coreference: ", str(end-start))
print(inputs)
print(coref)
