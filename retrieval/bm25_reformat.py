import os
import sys
import json


short2long = {"de": "german", "it": "italian", "fi": "finnish", "en": "english", "ru": "russian", "ha": "hausa", "sw": "swahili", "yo": "yoruba", "so": "somali"}
long2short = {v: k for k, v in short2long.items()}
def get_short(inpt):
    if len(inpt) == 2: return inpt
    return long2short[inpt]

def get_long(inpt):
    if len(inpt) > 2: return inpt
    return short2long[inpt]

def main():
    dataset = sys.argv[1] # choices: "clef", "ciral"
    qlang = sys.argv[2] # query language, e.g. "german" (clef) or "hausa" (ciral)
    dlang = sys.argv[3] 
    translate = sys.argv[4] # choices: QT, DT, OG

    docs_folder = "docs" if translate == "OG" else "docs_translation"

    if dataset == "clef":
        docs_filename = f"nllb_sentence_{get_short(qlang)}2{get_short(dlang)}.jsonl" if translate == "DT" else f"{get_long(dlang)}.jsonl"
    else:
        assert dataset == "ciral"
        docs_filename = get_long(dlang) + ".jsonl"

    script_path = os.path.dirname(os.path.abspath(__file__))
    fPath = f"{script_path}/../data/{dataset}/{docs_folder}/{docs_filename}"
    
    lines = []
    print(f"Reading {fPath}")
    with open(fPath, "r") as f:
        for line in f:
            r = json.loads(line)
            if dataset == "clef":
                lines.append({"id": str(r["docid"]), "contents": r["text"]})
            else:
                assert dataset == "ciral"
                lines.append({"id": str(r["docid"]), "contents": r["title"] + " " + r["text"]})
    
    folder = f"{script_path}/indexes/{dataset}/{get_short(qlang)}2{get_short(dlang)}"
    fPath = f"{folder}/docs.jsonl"
    print(f"Writing {len(lines)} '{get_long(dlang)}' documents translated into '{get_long(qlang)}': {fPath}")
    os.makedirs(folder, exist_ok=True)
    with open(fPath, "w") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()
