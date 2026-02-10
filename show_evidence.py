from src.modules.datasets.feverous import FeverousEvidenceFormat
import json

dataset = FeverousEvidenceFormat.from_path("/raid/Workspace/an/code/factcheck/FEVEROUS/data/dev.jsonl",
                             "/raid/Workspace/an/code/factcheck/FEVEROUS/data/feverous_wikiv1.db")

n = 0
for sample in dataset:
    print('--------------------------------')
    print('claim : ', sample['claim'])
    # print('context : ', sample['context'])
    print('evidence : \n')
    print(sample['evidence'])
    print('label : ', sample['label'])
    print('--------------------------------\n\n')
    n += 1
    if n > 100:
        break
