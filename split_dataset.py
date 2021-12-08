import json
from collections import defaultdict
import random


def split_dataset(dataset):
    test = defaultdict(list)
    train = defaultdict(list)
    for topic, papers in dataset.items():
        random.shuffle(papers)
        train_papers, test_papers = papers[10:], papers[:10]
        train[topic].extend(train_papers)
        test[topic].extend(test_papers)
    return train, test


if __name__ == "__main__":
    random.seed(0)
    with open("./dataset.json") as f:
        dataset = json.load(f)
    train, test = split_dataset(dataset)

    with open("./train.json", "w") as f:
        json.dump(train, f, indent=4)

    with open("./test.json", "w") as f:
        json.dump(test, f, indent=4)