import json
import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

CIRAL_LANGUAGES = [
    "hausa",
    "yoruba",
    "swahili",
    "somali"
]
CIRAL_SPLIT = 'testA'

CLEF_LANGUAGES = [
    "english", 
    "finnish",
    "german",
    "italian",
    "russian"
]

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
load_dotenv(ENV_PATH)
DATA_LOCATION = Path(os.getenv("DATA_LOCATION"))

def normalize_whitespace(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def load_ciral():
    for language in tqdm(CIRAL_LANGUAGES):
        print(f"Processing CIRAL data for {language}")
        ciral_dataset = None

        # Extract documents
        for translated in [True, False]:
            doc_file = f"{language}.jsonl"
            folder = "docs" if not translated else "docs_translation"
            doc_path = DATA_LOCATION / "ciral" / folder / doc_file
            
            if os.path.exists(doc_path):
                print(f"Skipping, file exists {doc_path}")
                continue
            
            ciral_corpus = load_dataset("ciral/ciral-corpus", language, translated=translated, trust_remote_code=True)
            print(f"Writing {doc_path}")
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            with open(doc_path, "w", encoding="utf-8") as f:
                for item in ciral_corpus["train"]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Extract queries
        queries_file = f"{language}.jsonl"
        queries_path = DATA_LOCATION / "ciral" / "queries" / queries_file
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(queries_path):
            print(f"Writing {queries_path}")
            with open(queries_path, "w", encoding="utf-8") as f:
                # lazy loading
                if ciral_dataset is None:
                    ciral_dataset= load_dataset("ciral/ciral", language, trust_remote_code=True)
                for data in ciral_dataset[CIRAL_SPLIT]:
                    try:
                        query_id = data['query_id']
                        query_text = data['query']
                        
                        query_obj = {
                            "qid": query_id,
                            "text": query_text
                        }
                        
                        f.write(json.dumps(query_obj, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"Warning: Skipping query (query: {query_id}) for language {language} due to error: {e}")
                        continue
        else:
            print(f"Skipping, file exists {queries_path}")
        
        # Extract qrels
        qrels_file = f"{language}.txt"
        qrels_path = DATA_LOCATION / "ciral" / "qrels" / qrels_file
        if not os.path.exists(qrels_path):
            # lazy loading
            if ciral_dataset is None:
                ciral_dataset= load_dataset("ciral/ciral", language, trust_remote_code=True)
            qrels_path.parent.mkdir(parents=True, exist_ok=True)
            with open(qrels_path, "w", encoding="utf-8") as f:
                for data in ciral_dataset[CIRAL_SPLIT]:
                    query_id = data['query_id']
                    
                    # Process positive passages (relevant = 1)
                    for qrel in data.get('pools_positive_passages', []):
                        docid = qrel['docid']
                        f.write(f"{query_id} 0 {docid} 1\n")
                    
                    for qrel in data.get('pools_negative_passages', []):
                        docid = qrel['docid']
                        f.write(f"{query_id} 0 {docid} 0\n")
        else:
            print(f"Skipping, file exists {qrels_path}")

def load_clef():
    from clef_dataloaders.clef_dataloader import load_documents, load_queries, load_relevance_assessments
    for language in tqdm(CLEF_LANGUAGES):
        print(f"Processing CLEF data for {language}")
        
        # Extract documents (from CLEF 2003)
        doc_file = f"{language}.jsonl"
        doc_path = DATA_LOCATION / "clef" / "docs" / doc_file
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        ids, docs = load_documents(language, "2003")
        with open(doc_path, "w", encoding="utf-8") as f:
            for doc_id, doc_text in enumerate(zip(ids, docs)):
                try:
                    doc_obj = {
                        "docid": doc_id,
                        "text": normalize_whitespace(doc_text)
                    }
                    f.write(json.dumps(doc_obj, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Warning: Skipping document (docid: {doc_id}) for language {language} due to error: {e}")
                    continue
        
        # Extract queries (from CLEF 2003, with descriptions)
        queries_file = f"{language}.jsonl"
        queries_path = DATA_LOCATION / "clef" / "queries" / queries_file
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        
        ids, queries = load_queries(language, "2003", include_desc=True)
        with open(queries_path, "w", encoding="utf-8") as f:
            for query_id, query_text in zip(ids, queries):
                try:
                    query_obj = {
                        "qid": query_id,
                        "text": query_text
                    }
                    f.write(json.dumps(query_obj, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Warning: Skipping query (query: {query_id}) for language {language} due to error: {e}")
                    continue
        
        # Extract relevance assessments (from CLEF 2003)
        qrels_file = f"{language}.txt"
        qrels_path = DATA_LOCATION / "clef" / "qrels" / qrels_file
        qrels_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(qrels_path, "w") as f:
            relevant_docs = load_relevance_assessments(language, "2003", load_non_relevant_docs=False)
            non_relevant_docs = load_relevance_assessments(language, "2003", load_non_relevant_docs=True)
            all_query_ids = set(relevant_docs.keys()) | set(non_relevant_docs.keys())
        
            for query_id in sorted(all_query_ids):
                # Write relevant documents first
                if query_id in relevant_docs:
                    for doc_id in relevant_docs[query_id]:
                        f.write(f"{query_id} 0 {doc_id} 1\n")
                
                # Then write non-relevant documents
                if query_id in non_relevant_docs:
                    for doc_id in non_relevant_docs[query_id]:
                        f.write(f"{query_id} 0 {doc_id} 0\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ciral", "clef", "both"], default="both",
                        help="Which dataset to load (default: both)")
    
    args = parser.parse_args()
    
    if args.dataset in ["ciral", "both"]:
        print("Loading CIRAL data")
        load_ciral()
    
    if args.dataset in ["clef", "both"]:
        print("Loading CLEF data")
        load_clef()

if __name__ == "__main__":
    main()