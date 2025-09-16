import os
import pickle
import faiss
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)
DATA_LOCATION = os.getenv("DATA_LOCATION")
RETRIEVAL_BASE = os.getenv("RETRIEVAL_BASE")

TOP_K = 100

M3_IDENTIFIER = "m3"
E5_IDENTIFIER = "e5"
E5_INSTRUCT_IDENTIFIER = "e5_instruct"
MGTE_IDENTIFIER = "mgte"
NVEMBED_IDENTIFIER = "nv"

EMB_DIMS = {
    M3_IDENTIFIER: 1024,
    E5_IDENTIFIER: 1024,
    E5_INSTRUCT_IDENTIFIER: 1024,
    MGTE_IDENTIFIER: 768,
    NVEMBED_IDENTIFIER: 4096,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qry_lang",type=str,required=True)
    parser.add_argument("--doc_lang",type=str,required=True)
    parser.add_argument("--encoder", type=str, choices=[M3_IDENTIFIER, E5_IDENTIFIER, E5_INSTRUCT_IDENTIFIER, MGTE_IDENTIFIER, NVEMBED_IDENTIFIER], required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    out_dir = os.path.join(RETRIEVAL_BASE, "rankings", args.dataset, args.encoder)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "clef":
        trec_filepath = os.path.join(out_dir, f"{args.qry_lang}_{args.doc_lang}.trec")
    else:
        trec_filepath = os.path.join(out_dir, f"{args.doc_lang}.trec")

    if os.path.exists(trec_filepath):
        print(f"Skipping, file exists: {trec_filepath}")
        exit(0)

    encoded_docs = os.path.join(os.path.join(RETRIEVAL_BASE, "encodings", args.dataset, args.encoder),
                        f"{args.dataset}_{args.doc_lang}_docs.pkl")
    print(f"Loading {encoded_docs}")
    with open(encoded_docs, 'rb') as f:
        doc_reps, doc_lookup = pickle.load(f)

    # ciral files use document language
    encoded_queries = os.path.join(os.path.join(RETRIEVAL_BASE, "encodings", args.dataset, args.encoder),
                        f"{args.dataset}_{args.qry_lang}_queries.pkl")
    print(f"Loading {encoded_queries}")
    with open(encoded_queries, 'rb') as f:
        query_reps, query_lookup = pickle.load(f)
    
    print("Running Retrieval")
    index = faiss.IndexFlatIP(EMB_DIMS[args.encoder])
    index.add(doc_reps)
    scores, indices = index.search(query_reps, TOP_K)

    
    print(f"Writing trec file: {trec_filepath}")
    with open(trec_filepath, "w") as f:
        for query_index, query_id in tqdm(enumerate(query_lookup), total=len(query_lookup), desc="Processing queries"):
            for rank, doc_index in enumerate(indices[query_index]):
                doc_id = doc_lookup[doc_index]
                score = scores[query_index][rank]
                f.write(f"{query_id} Q0 {doc_id} {rank+1} {score} dense\n")
                f.flush()

if __name__ == "__main__":
    main()