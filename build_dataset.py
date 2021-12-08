import json
import re
import urllib.request as libreq
from collections import defaultdict

from bs4 import BeautifulSoup
from tqdm import tqdm


def add_query(dataset, soup, topic):
    try:
        summary = re.sub(r"<.*?>", "", str(soup.find("summary"))).strip().replace("\n", " ")
        # label = re.sub(r"<.*?>", "", str(soup.find("arxiv:primary_category").attrs["term"])).strip()
        title = re.sub(r"<.*?>", "", str(soup.findAll("title")[1])).strip().replace("\n", " ")
        dataset[topic].append({"title": title, "summary": summary})
    except:
        print(soup.prettify())

if __name__ == "__main__":
    topics = [
        "artificial+intelligence",
        "hardware+architecture",
        "computer+security",
        "databases",
        "formal+languages+and+automata+theory",
        "operating+systems",
        "computer+networking",
        "numerical+analysis"
    ]
    dataset = defaultdict(list)
    num_papers = 100

    iter = tqdm(topics, total=len(topics))
    for topic in iter:
        iter.set_description(f"getting papers for {topic}")
        for i in range(num_papers):
            with libreq.urlopen(f'http://export.arxiv.org/api/query?search_query=all:{topic}&start={i}&max_results=1') as url:
                html_doc = url.read().decode("utf-8")

            soup = BeautifulSoup(html_doc, 'html.parser')
            add_query(dataset, soup, topic)

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    # print(json.dumps(dataset, indent=4))