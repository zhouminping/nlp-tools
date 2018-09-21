import json


def to_json_dict_str(inputs: str) -> str:
    inputs = {"sentence": inputs.strip()}
    return json.dumps(inputs)

