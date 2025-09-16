#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR

# clef
echo "Running bm25 on CLEF"
bash run_bm25.sh clef en fi
bash run_bm25.sh clef en it
bash run_bm25.sh clef en ru
bash run_bm25.sh clef en de
bash run_bm25.sh clef de fi
bash run_bm25.sh clef de it
bash run_bm25.sh clef de ru
bash run_bm25.sh clef fi it
bash run_bm25.sh clef fi ru

# ciral
echo "Running bm25 on CIRAL"
bash run_bm25.sh ciral en ha
bash run_bm25.sh ciral en so
bash run_bm25.sh ciral en sw
bash run_bm25.sh ciral en yo
