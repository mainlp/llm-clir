import logging
import llmrankers.pairwise as pw  
from transformers import AutoTokenizer

pw.T5Tokenizer = AutoTokenizer

from llmrankers.pairwise import PairwiseLlmRanker
from llmrankers.rankers import SearchResult
# from huggingface_hub import login
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
import os

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error() 

random.seed(929)
logger = logging.getLogger(__name__)


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\trun\n")
                rank += 1


def main(args):
    if args.pairwise:
        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
            logger.info(f'Setting batch_size to 2.')


        ranker = PairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       method=args.pairwise.method,
                                       batch_size=args.pairwise.batch_size,
                                       k=args.pairwise.k)

        ranker.tokenizer.pad_token = ranker.tokenizer.eos_token

    else:
        raise ValueError('Choose pairwise')

    query_map = {}
    if args.run.query_file:
        with open(args.run.query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qid = None
                query_text = None
                
                obj = json.loads(line)
                qid = obj.get("qid")
                query_text = obj.get("text")

                if qid is None or query_text is None:
                    continue
                qid = str(qid) 
                query_map[qid] = ranker.truncate(query_text, args.run.query_length)

    # logger.info(f'Loading first stage run from {args.run.run_path}.')
    
    first_stage_rankings = []
    
    with open(args.run.candidates_file, 'r', encoding='utf-8') as f:
        current_qid = None
        current_ranking = []
        
        for line in tqdm(f):
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue  

            qid = parts[0]  # query_id
            docid = parts[1]  # doc_id
            try:
                score = float(parts[2])
            except ValueError:
                print(f"Skipping line due to invalid score: {line.strip()}")
                continue
            doc_text = parts[3] 
            
            if qid != current_qid:
                if current_qid is not None:
                    cq = str(current_qid)
                    if cq in query_map:
                        first_stage_rankings.append((cq, query_map[cq], current_ranking[:args.run.hits]))
                    else:
                        logger.warning(f"qid {cq} not found in query_map; skipping this group.")
                current_ranking = []
                current_qid = qid
            
            if len(current_ranking) >= args.run.hits:
                continue 
            
            text = ranker.truncate(doc_text, args.run.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=score, text=text))

    if current_qid is not None:
        cq = str(current_qid)
        if cq in query_map:
            first_stage_rankings.append((cq, query_map[cq], current_ranking[:args.run.hits]))
        else:
            logger.warning(f"qid {cq} not found in query_map; skipping last group.")
            
    reranked_results = []
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings):
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                pass
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
        reranked_results.append((qid, query, ranker.rerank(query, ranking)))
        total_comparisons += ranker.total_compare
        total_prompt_tokens += ranker.total_prompt_tokens
        total_completion_tokens += ranker.total_completion_tokens
    toc = time.time()

    print(f'Avg comparisons: {total_comparisons/len(reranked_results)}')
    print(f'Avg prompt tokens: {total_prompt_tokens/len(reranked_results)}')
    print(f'Avg completion tokens: {total_completion_tokens/len(reranked_results)}')
    print(f'Avg time per query: {(toc-tic)/len(reranked_results)}')

    write_run_file(args.run.save_path, reranked_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')


    run_parser = commands.add_parser('run')
    # run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--query_file', type=str, default=None)
    run_parser.add_argument('--candidates_file', type=str, default=None, help="filepath for tsv input file (generated with process_pairwise.py)")
    # run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=256)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    # run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument('--dataset', type=str, default="ciral", choices=['clef', 'ciral'])

    pairwise_parser = commands.add_parser('pairwise')
    pairwise_parser.add_argument('--method', type=str, default='allpair',
                                 choices=['allpair', 'heapsort', 'bubblesort'])
    pairwise_parser.add_argument('--batch_size', type=int, default=2)
    pairwise_parser.add_argument('--k', type=int, default=10)
    


    args = parse_args(parser, commands)
    # print("DEBUG: args.pairwise.k =", args.pairwise.k)

    # if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
    #     raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    if os.path.exists(args.run.save_path):
        print(f"Skipping, file exists: {args.run.save_path}")
        exit(0)

    arg_dict = vars(args)
    if arg_dict['run'] is None or sum(arg_dict[arg] is not None for arg in arg_dict) != 2:
        raise ValueError('Need to set --run and can only set one of --pointwise, --pairwise, --setwise, --listwise')
    main(args)