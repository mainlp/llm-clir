import argparse
import json
import os
from collections import defaultdict

LANG_ABBR_TO_FULL = {
    "fi": "finnish",
    "de": "german",
    "it": "italian",
    "ru": "russian",
    "en": "english"
}
FULL_TO_LANG_ABBR = {v: k for k, v in LANG_ABBR_TO_FULL.items()}

def load_doc_texts(doc_path):
    doc_texts = {}
    with open(doc_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_texts[doc["docid"]] = {"content": doc["text"]}
    return doc_texts


def load_queries(query_path):
    qid_to_query = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            qid_to_query[str(q["qid"])] = {
                "qid": str(q["qid"]),
                "text": q["text"]
            }
    return qid_to_query


def load_ranking_trec(ranking_path):
    grouped = defaultdict(list)
    with open(ranking_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                qid, _, docid, _, score, _ = parts
            elif len(parts) == 4:
                qid, _, docid, rank = parts
                score = float(1000 - int(rank))  # fake score
            grouped[qid].append({
                "docid": docid,
                "score": float(score)
            })
    return grouped


def generate_jsonl(ranking_path, doc_path, query_path, output_path):
    grouped = load_ranking_trec(ranking_path)
    doc_texts = load_doc_texts(doc_path)
    qid_to_query = load_queries(query_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for qid, candidates in grouped.items():
            for c in candidates:
                doc_content = doc_texts.get(c["docid"], {"content": ""})["content"]
                c["doc"] = {"content": doc_content}
            query_info = qid_to_query.get(str(qid), {"qid": str(qid), "text": ""})
            record = {"query": query_info, "candidates": candidates}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input JSONL format for listwise reranking.")

    parser.add_argument("--input_dir", type=str, required=True, help="Dataset base dir with docs and queries")
    parser.add_argument("--ranking_dir", type=str, required=True, help="Dir containing .trec ranking files")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to write .jsonl files")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix (e.g. 'yoruba' or 'en2de')")
    parser.add_argument("--dataset", type=str, required=True, help="Options: 'clef' or 'ciral'")
    parser.add_argument("--translation", type=str, required=True, help="Options: 'OG' or 'DT'", choices=["DT", "OG"])

    args = parser.parse_args()

    prefix = args.prefix

    dataset = args.dataset
    qlang, dlang = prefix.split("_") if dataset == "clef" else (prefix, prefix)


    if args.translation == "DT" and args.dataset == "clef":
        # documents were translated sentence by sentence with nllb from the tgt language to the src language
        doc_path = os.path.join(args.input_dir, "docs_translation", f"nllb_sentence_{FULL_TO_LANG_ABBR[qlang]}2{FULL_TO_LANG_ABBR[dlang]}.jsonl")
    else:
      doc_path = os.path.join(args.input_dir, "docs" if args.translation == "OG" else "docs_translation", f"{dlang}.jsonl")

    output_path = os.path.join(args.output_dir, f"{prefix}.jsonl")
    ranking_path = os.path.join(args.ranking_dir, f"{prefix}.trec")

    if os.path.exists(output_path):
        print(f"Skipping, file exists {output_path}")
        exit(0)
    
    DATA_LOCATION = os.getenv("DATA_LOCATION", "")
    OUT_BASE = os.getenv("OUT_BASE", "")
    
    input_dir = args.input_dir or (os.path.join(DATA_LOCATION, dataset) if DATA_LOCATION else None)
    ranking_dir = args.ranking_dir or (
        os.path.join(OUT_BASE, "retrieval", "ranking", dataset, "BM25") if OUT_BASE else None
    )
    output_dir = args.output_dir or (
        os.path.join(OUT_BASE, dataset, "processed_listwise") if OUT_BASE else None
    )
    
    missing = []
    if not input_dir:  missing.append("--input_dir or $DATA_LOCATION")
    if not ranking_dir: missing.append("--ranking_dir or $OUT_BASE")
    if not output_dir: missing.append("--output_dir or $OUT_BASE")
    if missing:
        raise SystemExit("Missing required paths. Provide CLI args or set them via .env / environment:\n  - " +
                         "\n  - ".join(missing))

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Missing document file: {doc_path}")
    if not os.path.exists(ranking_path):
        raise FileNotFoundError(f"Missing ranking file: {ranking_path}")
    query_path = os.path.join(args.input_dir, "queries", f"{qlang}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    generate_jsonl(ranking_path=ranking_path, doc_path=doc_path, query_path=query_path, output_path=output_path)
