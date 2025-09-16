import os
import json
import sys
from bm25_reformat import get_long
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

dataset = sys.argv[1] # 'clef' or 'ciral'
qlang = sys.argv[2] # e.g. 'en' or 'english
dlang = sys.argv[3] # e.g. 'de' or 'german'
translation = sys.argv[4] # OG vs. DT
index_dir = sys.argv[5]

# ciral only has english queries, query files have the same name as document files (different queries for different languages).
# in clef, query files are named after the query language (e.g., "english.jsonl"; queries are parallel across different languages).
filename = get_long(qlang) if dataset == "clef" else get_long(dlang)

qids, queries = [], []
script_path = os.path.dirname(os.path.abspath(__file__))
with open(f"{script_path}/../data/{dataset}/queries/{filename}.jsonl") as f:
    for l in f:
        r = json.loads(l)
        qids.append(r["qid"])
        queries.append(r["text"])

results_dir = f"{script_path}/rankings/{dataset}/bm25"
os.makedirs(results_dir, exist_ok=True)
if dataset == "clef":
    fPath = os.path.join(results_dir, f"{get_long(qlang)}_{get_long(dlang)}.trec")
else:
    assert dataset == "ciral"
    fPath = os.path.join(results_dir, f"{get_long(dlang)}.trec")

n_queries = len(qids)
print(f"Writing bm25 run file {fPath} containing {n_queries} queries")
with open(fPath, "w") as f:
    lucene_bm25_searcher = LuceneSearcher(index_dir)
    lucene_bm25_searcher.set_language(qlang)
    for qid, query in tqdm(zip(qids, queries), total=n_queries):
        k = 100
        hits = lucene_bm25_searcher.search(query, k=k)
        for i in range(0, min(len(hits), k)):
            # print(f'{i + 1:2} {hits[i].docid:7} {hits[i].score:.5f}')
            f.write(f"{qid}\tQ0\t{hits[i].docid}\t{i+1}\t{hits[i].score:.5f}\tBM25\n")
