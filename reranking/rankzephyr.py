import argparse
import os
import sys
from pathlib import Path
from glob import glob
from dotenv import load_dotenv

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH)

PROJECT_ROOT     = Path(os.getenv("PROJECT_ROOT", REPO_ROOT))
DATA_LOCATION    = Path(os.getenv("DATA_LOCATION", PROJECT_ROOT / "data"))
RERANKING_BASE   = Path(os.getenv("RERANKING_BASE", PROJECT_ROOT / "reranking"))

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.rerank.listwise import ZephyrReranker

def process_file(input_path, output_dir, reranker):
    base_name = Path(input_path).stem
    output_path = output_dir / f"{base_name}.trec"
    
    if os.path.exists(output_path):
        print(f"File exists, skipping {output_path}")
        return
    
    print(f"Processing: {input_path}")
    
    requests = read_requests_from_file(input_path)
    
    # src: https://github.com/castorini/rank_llm?tab=readme-ov-file#-quick-start
    # kwargs = {"populate_invocations_history": True}
    # rerank_results = reranker.rerank_batch(requests=requests, **kwargs)
    rerank_results = reranker.rerank_batch(requests=requests)

    output_dir.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(rerank_results)
    writer.write_in_trec_eval_format(output_path)
    # writer.write_in_jsonl_format(output_dir / f"{base_name}_zephyr.jsonl")
    # writer.write_in_trec_eval_format(output_dir / f"{base_name}_zephyr_output.txt")
    # writer.write_inference_invocations_history(output_dir / f"{base_name}_zephyr.json")
    
def main():
    parser = argparse.ArgumentParser(description="Batch listwise reranking with ZephyrReranker")
    parser.add_argument("--dataset", type=str, required=True, choices=["clef", "ciral"],
                        help="Dataset name: clef or ciral")
    parser.add_argument("--retriever", type=str, required=True,
                        help="Subdirectory under processed_listwise/ (e.g., BM25_OG)")
    args = parser.parse_args()  

    input_dir = RERANKING_BASE / "processed_listwise" / args.dataset / args.retriever

    retriever, og_or_dt = args.retriever.split("_", 1)
    reranker_sub = f"Rankzephyr_{og_or_dt}"

    output_dir = RERANKING_BASE / "rankings" / "listwise" / args.dataset / retriever / reranker_sub

    jsonl_files = glob(os.path.join(input_dir, "*.jsonl"))
    reranker = ZephyrReranker()

    for input_file in jsonl_files:
        process_file(input_file, output_dir, reranker)
    print(f"All files processed. Results saved to {output_dir}")
    
if __name__ == "__main__":
    main()