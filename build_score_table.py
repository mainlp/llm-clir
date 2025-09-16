#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import numpy as np
from ir_measures import *
from ir_measures import read_trec_run, read_trec_qrels, iter_calc
from scipy.stats import ttest_rel

from ir_measures import MAP, nDCG, read_trec_run, read_trec_qrels, calc_aggregate

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH)

PROJECT_ROOT   = Path(os.getenv("PROJECT_ROOT")).resolve()
DATA_LOCATION  = Path(os.getenv("DATA_LOCATION"))
PRERANKING_BASE = Path(os.getenv("RETRIEVAL_BASE"))
RERANKING_BASE  = Path(os.getenv("RERANKING_BASE"))

all_langs = {
    "en": "english",
    "de": "german",
    "fi": "finnish",
    "it": "italian",
    "ru": "russian",
    "ha": "hausa",
    "so": "somali",
    "sw": "swahili",
    "yo": "yoruba"
}
# Metric mapping
metric_map = {
    "MAP": MAP,
    "nDCG@20": nDCG@20
}

# Dataset language definitions
dataset_langs = {
    "ciral": ["hausa", "somali", "swahili", "yoruba"],
    #"clef": ["en2fi", "en2it", "en2ru", "en2de", "de2fi", "de2it", "de2ru", "fi2it", "fi2ru"]
    "clef": ["english_finnish", "english_italian", "english_russian", "english_german", "german_finnish", "german_italian", "german_russian", "finnish_italian", "finnish_russian"]
}

long2short = {v: k for k, v in all_langs.items()}

# Normalize model name
def normalize_reranker(name):
    if "Rankzephyr" in name:
        return "RankZephyr (OG)" if "OG" in name else "RankZephyr (DT)"
    elif name.startswith("3_5"):
        return "RankGPT3.5 (OG)" if "OG" in name else "RankGPT3.5 (DT)"
    elif name.startswith("4_1"):
        return "RankGPT4.1 (OG)" if "OG" in name else "RankGPT4.1 (DT)"
    elif name.startswith("llama_3_1"):
        return "Llama3.1-8B-Instruct (OG)" if "OG" in name else "Llama3.1-8B-Instruct (DT)"
    elif name.startswith("aya-101"):
        return "Aya-101 (OG)" if "OG" in name else "Aya-101 (DT)"
    else:
        return name

def evaluate_with_stats(run_path, qrels_path, metric):
    qrels = read_trec_qrels(qrels_path)
    run = read_trec_run(run_path)
    results = calc_aggregate([metric], qrels, run)
    for _, val in results.items():
        return round(val, 3)

def langpair_heading(lang_or_langpair, dataset):
    if dataset == "clef":
        qlang, dlang = lang_or_langpair.split("_")
        return long2short.get(qlang, qlang) + "2" + long2short.get(dlang, dlang)
    else:
        return "en2" + long2short.get(lang_or_langpair, lang_or_langpair)


def calculate_score(metric, qrels_path: Path, run_path: Path):
    qrels = list(read_trec_qrels(str(qrels_path)))
    run = list(read_trec_run(str(run_path)))
    vals = iter_calc([metric], qrels, run)
    return {v.query_id: float(v.value) for v in vals}

def metric_for_ttest(dataset: str):
    return AP if dataset == "clef" else (nDCG@20)

def t_test(metric, qrels_path: Path, rerank_run: Path, prerank_run: Path) -> str:
    if (not rerank_run.exists()) or (not prerank_run.exists()):
        print(f"Missing run files.")
        return ""

    rerank = calculate_score(metric, qrels_path, rerank_run)
    prerank = calculate_score(metric, qrels_path, prerank_run)
    if not rerank or not prerank:
        print(f"No scores calculated for {rerank_run} and {prerank_run}")
        return ""

    common_ids = sorted(set(rerank.keys()) & set(prerank.keys()))
    if not common_ids:
        print(f"No common query IDs between {rerank_run} and {prerank_run}")
        return ""

    rerank_vals = np.array([rerank[i] for i in common_ids], dtype=float)
    prerank_vals = np.array([prerank[i] for i in common_ids], dtype=float)

    t_stat, p_val = ttest_rel(rerank_vals, prerank_vals, nan_policy="omit")
    if np.isnan(t_stat) or np.isnan(p_val):
        print(f"NaN encountered in t-test for {rerank_run} and {prerank_run}")
        return ""

    if (p_val < 0.05) and (rerank_vals.mean() > prerank_vals.mean()):
        return "*"
    return ""

def run_evaluation(stage: str, dataset: str, approach: str, output_dir: Path, ttest: bool):
    print(f"Evaluating {stage} results")
    print(f"Dataset: {dataset}")
    print(f"Output directory: {output_dir}")
    print("-------------------------------------")

    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "clef":
        qrels_dir = DATA_LOCATION / dataset / "qrels"
        metric = metric_map["MAP"]
        langs = dataset_langs["clef"]
    elif dataset == "ciral":
        qrels_dir = DATA_LOCATION / dataset / "qrels"
        metric = metric_map["nDCG@20"]
        langs = dataset_langs["ciral"]

    table = defaultdict(lambda: defaultdict(str))

    if stage == "reranking":
        results_dir = RERANKING_BASE / "rankings" / approach / dataset
        print(f"Reranking results directory: {results_dir}")
        print(f"Qrels directory: {qrels_dir}")
        output_file = output_dir / f"{dataset}_{approach}_table.tsv"

        for retriever in sorted(results_dir.glob("*")):
            if not retriever.is_dir():
                continue
            for reranker in sorted(retriever.glob("*")):
                if not reranker.is_dir():
                    continue
                print(f"Processing: {retriever.name} - {reranker.name}")
                key = (normalize_reranker(reranker.name), retriever.name)
                vals = []

                for lang_or_langpair in langs:
                    trec_name = f"{lang_or_langpair}.trec"
                    run_path = reranker / trec_name  

                    if dataset == "ciral":
                        qrels_path = qrels_dir / f"{lang_or_langpair}.txt"
                    else:
                        target_lang = lang_or_langpair.split("_")[-1].lower()
                        qrels_path = qrels_dir / f"{target_lang}.txt"

                    result = evaluate_with_stats(str(run_path), str(qrels_path), metric)

                    prerank_run = PRERANKING_BASE / "rankings" / dataset / retriever.name / trec_name
                    
                    if ttest:
                        sig_metric = metric_for_ttest(dataset)
                        sig_mark = t_test(sig_metric, qrels_path, run_path, prerank_run) if result is not None else ""
                    else:
                        sig_mark = ""

                    if result is not None:
                        rerank_result = f"{result:.3f}{sig_mark}"
                        table[key][langpair_heading(lang_or_langpair=lang_or_langpair, dataset=dataset)] = rerank_result
                        vals.append(result)

                if vals:
                    table[key]["AVG"] = round(np.mean(vals), 3)

        df = pd.DataFrame.from_dict(table, orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Reranker", "Retriever"])
        df = df.reset_index()
        df['retriever_rank'] = df['Retriever'].apply(lambda x: 1 if x.lower() == 'bm25' else 0)

        def get_reranker_base(name):
            if "Zephyr" in name:
                return 0
            elif "3.5" in name:
                return 1
            elif "4.1" in name:
                return 2
            elif "3.1" in name:
                return 3
            elif "101" in name:
                return 4
            return 5

        df['reranker_base'] = df['Reranker'].apply(get_reranker_base)
        df['og_rank'] = df['Reranker'].apply(lambda x: 0 if '(OG)' in x else 1)
        df = df.sort_values(by=['retriever_rank', 'reranker_base', 'og_rank', 'Retriever', 'Reranker'])
        df = df.drop(columns=['retriever_rank', 'reranker_base', 'og_rank'])
        df = df.set_index(['Reranker', 'Retriever'])
        df.to_csv(output_file, sep="\t")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        print(df)
        print(f"Results saved to: {output_file}")
    
    else: 
        results_dir = PRERANKING_BASE / "rankings" / dataset
        print(f"Preranking results directory: {results_dir}")
        print(f"Qrels directory: {qrels_dir}")
        output_file = output_dir / f"{dataset}_preranking_table.tsv"

        for retriever in sorted(results_dir.glob("*")):
            if not retriever.is_dir():
                continue
            print(f"Processing: {retriever.name}")
            key = retriever.name  
            vals = []

            for lang_or_langpair in langs:
                trec_name = f"{lang_or_langpair}.trec"
                run_path = retriever / trec_name

                if dataset == "ciral":
                    qrels_path = qrels_dir / f"{lang_or_langpair}.txt"
                else:
                    target_lang = lang_or_langpair.split("_")[-1].lower()
                    qrels_path = qrels_dir / all_langs.get(target_lang, f"{target_lang}.txt")

                result = evaluate_with_stats(str(run_path), str(qrels_path), metric)
                
                if result is not None:
                    table[key][langpair_heading(lang_or_langpair=lang_or_langpair, dataset=dataset)] = f"{result:.3f}"
                    vals.append(result)

            if vals:
                table[key]["AVG"] = round(np.mean(vals), 3)

        df = pd.DataFrame.from_dict(table, orient="index")
        df.index.name = "Retriever"
        df = df.sort_index()
        df.to_csv(output_file, sep="\t")
        print(df)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reranking for CLEF or CIRAL datasets.")
    parser.add_argument("--stage", type=str, required=True, choices=["retrieval", "reranking"],
                        help="retrieval or reranking.")
    parser.add_argument("--approach", type=str, choices=["listwise", "pairwise"],
                        help="listwise or pairwise.")
    parser.add_argument("--dataset", type=str, required=True, choices=["clef", "ciral"],
                        help="clef or ciral")
    parser.add_argument("--output_dir", type=str, default=PROJECT_ROOT,
                        help="Directory to save the result table.")
    parser.add_argument('--ttest', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.stage == "reranking" and not args.approach:
        parser.error("'--approach' is required when evaluating reranking results.")

    run_evaluation(stage=args.stage, approach=args.approach, dataset=args.dataset, output_dir=Path(args.output_dir), ttest=args.ttest)