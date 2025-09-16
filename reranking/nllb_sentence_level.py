import argparse
import os
import sys
import json
import re
import io
import torch
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize

# Ensure UTF-8 default encoding for safety
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Download punkt only if missing (avoids repeated downloads)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def map_nltk_lang(sent_token_lang: str | None, src_lang: str | None) -> str | None:
    """
    Map user-provided --sent_token_lang or --src_lang (NLLB code) to NLTK language names.
    Returns an NLTK language string if supported, else None.
    """
    # If user already passed an nltk name, trust it.
    if sent_token_lang:
        low = sent_token_lang.strip().lower()
        if low in {"english", "german", "finnish", "italian", "russian"}:
            return low

    # Map common NLLB codes to NLTK names
    code = (src_lang or "").lower()
    mapping = {
        "eng_latn": "english",
        "deu_latn": "german",
        "fin_latn": "finnish",
        "ita_latn": "italian",
        "rus_cyrl": "russian",
    }
    return mapping.get(code, None)

_SENT_SPLIT_REGEX = re.compile(r'(?<=[\.!?…])\s+(?=[^\s])', flags=re.UNICODE)

def safe_split_into_sentences(text: str, nltk_lang: str | None) -> list[str]:
    """
    Try NLTK sentence tokenizer if nltk_lang is supported, otherwise fall back to a regex splitter.
    """
    text = (text or "").strip()
    if not text:
        return []
    if nltk_lang:
        try:
            return sent_tokenize(text, language=nltk_lang)
        except Exception:
            pass
    # Fallback: simple regex split; keep text intact if regex finds nothing
    parts = _SENT_SPLIT_REGEX.split(text)
    return parts if parts else [text]

def batchify(lst, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def batch_translate(
    sentences: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    batch_size: int,
    max_length: int,
    bos_token_id: int,
) -> list[str]:
    translated: list[str] = []
    for batch in batchify(sentences, batch_size):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=bos_token_id,
                num_beams=1,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)
    return translated

def read_processed_docids(out_path: str) -> set[str]:
    """
    For resume: read existing output JSONL and collect docids to skip.
    """
    seen = set()
    if not os.path.exists(out_path):
        return seen
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                docid = obj.get("docid")
                if isinstance(docid, str):
                    seen.add(docid)
            except Exception:
                # If a corrupted line exists, ignore and continue
                continue
    return seen

def main():
    parser = argparse.ArgumentParser(description="NLLB sentence-level translator for JSONL {docid, text}.")
    parser.add_argument('--input_file', required=True, help="JSONL file with objects: {'docid': str, 'text': str}")
    parser.add_argument('--output_file', required=True, help="JSONL output file with same keys: {'docid','text'} (translated)")
    parser.add_argument('--src_lang', required=True, help="NLLB code of source language, e.g., deu_Latn")
    parser.add_argument('--tgt_lang', default='eng_Latn', help="NLLB code of target language, e.g., eng_Latn")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--model_name', default='facebook/nllb-200-1.3B')
    parser.add_argument('--gpu', default='0', help="CUDA visible devices index, e.g., '0' or '0,1'")
    parser.add_argument('--sent_token_lang', default=None, help="NLTK language name (english/german/...) if you want to override")
    args = parser.parse_args()

    # Device setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
    ).to(device)

    # Set language directions
    tokenizer.src_lang = args.src_lang
    bos_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    # Pick NLTK lang (or None -> fallback splitter)
    nltk_lang = map_nltk_lang(args.sent_token_lang, args.src_lang)

    # Resume support: collect already processed docids
    processed = read_processed_docids(args.output_file)
    processed_count = len(processed)
    print(f"Resuming: {processed_count} doc(s) already in {args.output_file}, will skip them.", file=sys.stderr)

    # Stream input and write output
    total = 0
    # Pre-count total lines for a nicer tqdm bar (optional)
    try:
        with open(args.input_file, 'r', encoding='utf-8') as fin:
            total = sum(1 for _ in fin)
    except Exception:
        total = 0

    # Append-safe
    out_mode = 'a' if os.path.exists(args.output_file) else 'w'
    with open(args.input_file, 'r', encoding='utf-8') as fin, \
         open(args.output_file, out_mode, encoding='utf-8') as fout:

        pbar = tqdm(fin, desc="Translating", total=total)
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                # Skip malformed JSON lines
                continue

            docid = obj.get("docid")
            text = obj.get("text")
            if not isinstance(docid, str) or not isinstance(text, str):
                # Skip records missing required keys
                continue
            if docid in processed:
                # already done
                continue

            try:
                sentences = safe_split_into_sentences(text, nltk_lang)
                if sentences:
                    translated_sentences = batch_translate(
                        sentences,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        bos_token_id=bos_token_id,
                    )
                    translated_text = " ".join(translated_sentences)
                else:
                    translated_text = ""
            except Exception as e:
                translated_text = f"[Translation Error: {e}]"

            out_obj = {"docid": docid, "text": translated_text}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()
            processed.add(docid)

if __name__ == "__main__":
    main()
