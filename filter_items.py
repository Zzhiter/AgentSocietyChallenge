import json
from collections import defaultdict

input_file = '/data/zhangzhe/AgentSocietyChallenge/data_output/item.json'
output_file = 'filtered_items.json'

def filter_items(input_file, output_file, limit=10):
    filtered_items = defaultdict(list)

    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            source = item.get("source")
            if len(filtered_items[source]) < limit:
                filtered_items[source].append(item)

    with open(output_file, 'w') as f:
        for items in filtered_items.values():
            for item in items:
                f.write(json.dumps(item) + '\n')

    print(f"Filtered items have been saved to {output_file}")

filter_items(input_file, output_file)