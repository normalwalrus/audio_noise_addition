import json

def get_configs(path):

    f = open(path)
    data = json.load(f)

    return data