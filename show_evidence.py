from src.modules.datasets.feverous import Feverous


dataset = Feverous.from_path("datas/feverous/feverous_dev_challenges.jsonl",
                             "datas/feverous/feverous_wikiv1.db")

for sample in iter(dataset):
    print(sample)