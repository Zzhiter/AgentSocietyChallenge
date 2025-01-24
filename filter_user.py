import json
from collections import defaultdict

input_file = '/data/zhangzhe/AgentSocietyChallenge/data_output/user.json'
output_file = 'filter_user.json'

def filter_items(input_file, output_file, limit=10):
    filter_user = defaultdict(list)

    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            source = item.get("source")
            if len(filter_user[source]) < limit:
                filter_user[source].append(item)

    with open(output_file, 'w') as f:
        for items in filter_user.values():
            for item in items:
                f.write(json.dumps(item) + '\n')

    print(f"Filtered items have been saved to {output_file}")

filter_items(input_file, output_file)