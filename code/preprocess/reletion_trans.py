import json

path = 'IA.json'
new_dict = {}
data = json.load(open(path))
for key,value in data.items():
    for val in value:
        if val in new_dict.keys():
            new_dict[val].append(key)
        else:
            new_dict[val] = [key]

new_list_json = json.dumps(new_dict, indent=4)
open('AI.json', 'w', encoding='utf-8').write(new_list_json)

path = 'OI.json'
new_dict = {}
data = json.load(open(path))
for key,value in data.items():
    for val in value:
        if val in new_dict.keys():
            new_dict[val].append(key)
        else:
            new_dict[val] = [key]

new_list_json = json.dumps(new_dict, indent=4)
open('IO.json', 'w', encoding='utf-8').write(new_list_json)

path = '/UO_build.json'
new_dict = {}
data = json.load(open(path))
for key,value in data.items():
    for val in value:
        if val in new_dict.keys():
            new_dict[val].append(key)
        else:
            new_dict[val] = [key]

new_list_json = json.dumps(new_dict, indent=4)
open('OU_build.json', 'w', encoding='utf-8').write(new_list_json)