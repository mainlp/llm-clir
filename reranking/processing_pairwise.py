import argparse
import os
import json
import pandas as pd
from collections import defaultdict

LANG_ABBR_TO_FULL = {
    "fi": "finnish",
    "de": "german",
    "it": "italian",
    "ru": "russian",
    "en": "english"
}
FULL_TO_LANG_ABBR = {v: k for k, v in LANG_ABBR_TO_FULL.items()}

def sanitize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()

def load_doc_texts(doc_path):
    doc_texts = {}
    with open(doc_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_texts[doc["docid"]] = sanitize_text(doc.get("text", ""))
    return doc_texts

def load_ranking_trec(ranking_path):
    from collections import defaultdict
    grouped = defaultdict(list)
    with open(ranking_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 6:
                qid, _, docid, _, score, _ = parts
                try:
                    score = float(score)
                except ValueError:
                    score = 0.0
            elif len(parts) == 4:
                qid, _, docid, rank = parts
                try:
                    score = float(1000 - int(rank))
                except Exception:
                    score = 0.0
            else:
                continue
            grouped[qid].append((docid, float(score)))
    return grouped

def generate_tsv(ranking_path, doc_path, output_path):
    grouped = load_ranking_trec(ranking_path)
    doc_texts = load_doc_texts(doc_path)

    rows = []
    for qid, doc_score_list in grouped.items():
        for docid, score in doc_score_list:
            content = sanitize_text(doc_texts.get(docid, ""))
            rows.append([qid, docid, score, content])

    df = pd.DataFrame(rows, columns=["query_id", "doc_id", "score", "doc_content"])
    df.to_csv(output_path, sep="\t", index=False, lineterminator="\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input TSV files for pairwise reranking.")
    parser.add_argument("--input_dir", type=str, required=True, help="Dataset base dir with docs and queries")
    parser.add_argument("--ranking_dir", type=str, required=True, help="Dir containing .trec ranking files")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to write .tsv files")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix (e.g. 'yoruba' or 'en2de')")
    parser.add_argument("--dataset", type=str, required=True, help="Options: 'clef' or 'ciral'")
    parser.add_argument("--translation", type=str, required=True, help="Options: 'OG' or 'DT'", choices=["DT", "OG"])
    args = parser.parse_args()

    prefix = args.prefix

    dataset = args.dataset
    qlang, dlang = prefix.split("_") if dataset == "clef" else ("", prefix)

    if args.translation == "DT" and args.dataset == "clef":
        # documents were translated sentence by sentence with nllb from the tgt language to the src language
        doc_path = os.path.join(args.input_dir, "docs_translation", f"nllb_sentence_{FULL_TO_LANG_ABBR[qlang]}2{FULL_TO_LANG_ABBR[dlang]}.jsonl")
    else:
      doc_path = os.path.join(args.input_dir, "docs" if args.translation == "OG" else "docs_translation", f"{dlang}.jsonl")
      
    ranking_path = os.path.join(args.ranking_dir, f"{prefix}.trec")
    output_path = os.path.join(args.output_dir, f"{prefix}.tsv")
    
    if os.path.exists(output_path):
        print(f"Skipping, file exists {output_path}")
        exit(0)
      
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Missing document file: {doc_path}")
    if not os.path.exists(ranking_path):
        raise FileNotFoundError(f"Missing ranking file: {ranking_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    generate_tsv(ranking_path=ranking_path, doc_path=doc_path, output_path=output_path)
