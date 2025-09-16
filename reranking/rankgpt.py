import os
import sys
from pathlib import Path
import argparse
from glob import glob

from rank_llm.rerank import Reranker
from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.rerank.listwise import SafeOpenai
# from rank_llm.retrieve import Retriever
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH)

PROJECT_ROOT     = Path(os.getenv("PROJECT_ROOT", REPO_ROOT))
DATA_LOCATION    = Path(os.getenv("DATA_LOCATION", PROJECT_ROOT / "data"))
RERANKING_BASE   = Path(os.getenv("RERANKING_BASE", PROJECT_ROOT / "reranking"))

def model_name_to_dir(model_name: str) -> str:
    mapping = {
        "gpt-4.1": "4_1",
        "gpt-3.5-turbo": "3_5",
    }
    return mapping.get(model_name, model_name.replace(".", "_").replace("-", "_"))

def process_file(input_path, output_dir, reranker):
    print(f"Processing: {input_path}")
    requests = read_requests_from_file(str(input_path))
    kwargs = {"populate_invocations_history": True}
    rerank_results = reranker.rerank_batch(requests, **kwargs)

    base_name = Path(input_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(rerank_results)
    output_path = output_dir / f"{base_name}.trec"
    writer.write_in_trec_eval_format(output_path)
    # writer.write_in_jsonl_format(output_dir / f"{base_name}_zephyr.jsonl")
    # writer.write_in_trec_eval_format(output_dir / f"{base_name}_zephyr_output.txt")
    # writer.write_inference_invocations_history(output_dir / f"{base_name}_zephyr.json")

def main():
    parser = argparse.ArgumentParser(description="Batch listwise reranking with RankGPT")
    parser.add_argument("--dataset", type=str, required=True, choices=["clef", "ciral"],
                        help="Dataset name: clef or ciral")
    parser.add_argument("--retriever", type=str, required=True,
                        help="Subdirectory under processed_listwise/ (e.g., BM25_OG)")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="OpenAI model name (default: gpt-4.1)")
    args = parser.parse_args()

    input_dir = RERANKING_BASE / "processed_listwise" / args.dataset / args.retriever

    retriever, og_or_dt = args.retriever.split("_", 1)
    model_sub = model_name_to_dir(args.model)
    reranker_sub = f"{model_sub}_{og_or_dt}"

    output_dir = RERANKING_BASE / "rankings" / "listwise" / args.dataset / retriever / reranker_sub

    api_key = os.environ.get("OPENAI_API_KEY")
    jsonl_files = sorted(Path(input_dir).glob("*.jsonl"))

    model_coordinator = SafeOpenai(args.model, context_size=4096, keys=api_key)
    reranker = Reranker(model_coordinator)

    for input_file in jsonl_files:
        process_file(input_file, output_dir, reranker)
    
if __name__ == "__main__":
    main()

