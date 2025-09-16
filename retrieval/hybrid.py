from __future__ import annotations
from collections import defaultdict
import argparse
import sys
from typing import Dict, Tuple, Iterable


def read_trec_file(path: str) -> Dict[str, Dict[str, int]]:
    """
    Read a TREC-formatted run file.
    Returns: dict[query_id][doc_id] = rank
    """
    ranks = defaultdict(dict)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                # Expected: qid Q0 docid rank score tag
                if len(parts) < 6:
                    continue
                qid, _, did, r_str, _, _ = parts[:6]
                try:
                    rank = int(r_str)
                except ValueError:
                    continue
                ranks[qid][did] = rank
    except FileNotFoundError:
        print(f"[Error] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    return ranks


def fuse_hybrid_rrf(
    bm25_ranks: Dict[str, Dict[str, int]],
    dense_ranks: Dict[str, Dict[str, int]],
    topk: int,
) -> Iterable[Tuple[str, str, int, float]]:
    """
    Perform hybrid fusion and return results.
    Output format: (qid, docid, rank, score)
    score = 1.0 / fused_rank
    """
    all_qids = sorted(set(bm25_ranks.keys()) | set(dense_ranks.keys()), reverse=True)
    
    k = 60
    
    for qid in all_qids:
        bm = bm25_ranks.get(qid, {})
        de = dense_ranks.get(qid, {})
        all_dids = set(bm) | set(de)
        items = []
        for did in all_dids:
            score = 0
            
            r_bm = bm.get(did, 0)
            if r_bm > 0:
                score += 1 / (r_bm + k)
            
            r_de = de.get(did, 0)
            if r_de > 0:
                score += 1 / (r_de + k)
            #score = 1 / (k + r_bm) + 1 / (k + r_de)
            items.append((did, score))
        
        # Sort by sum of reciprocal ranks (descending).
        items.sort(key=lambda x: -x[1])
        
        for final_rank, (did, score) in enumerate(items[:topk], start=1):
            yield qid, did, final_rank, float(f"{score:.6f}")


def write_trec(rows: Iterable[Tuple[str, str, int, float]], path: str, tag: str) -> None:
    """
    Write results in TREC run format:
    qid Q0 docid rank score tag
    """
    with open(path, "w", encoding="utf-8") as f:
        for qid, did, rank, score in rows:
            f.write(f"{qid} Q0 {did} {rank} {score:.6f} {tag}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid rank fusion (BM25 + Dense) via weighted average ranks (TREC format only)."
    )
    p.add_argument("--bm25", required=True, help="Path to BM25 TREC run file")
    p.add_argument("--dense", required=True, help="Path to Dense TREC run file")
    p.add_argument("--output", required=True, help="Path to output TREC run file")
    p.add_argument("--default-rank", type=int, default=1001,
                   help="Default rank when a document is missing (default: 1001)")
    p.add_argument("--topk", type=int, default=100, help="Keep top K documents per query (default: 100)")
    p.add_argument("--tag", default="hybrid",
                   help="Run tag used in TREC output (default: hybrid)")
    return p.parse_args()


def main():
    args = parse_args()

    bm25_ranks = read_trec_file(args.bm25)
    dense_ranks = read_trec_file(args.dense)

    rows = list(
        fuse_hybrid_rrf(
            bm25_ranks=bm25_ranks,
            dense_ranks=dense_ranks,
            topk=args.topk,
        )
    )

    try:
        write_trec(rows, args.output, tag=args.tag)
    except Exception as e:
        print(f"[Error] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
