#!/bin/bash

# SimCSE-BERT
python evaluation.py \
    --model_name_or_path princeton-nlp/unsup-simcse-bert-base-uncased \
    --pooler hidden_states_subspace \
    --task_set sts \
    --mode test

# BERT
python evaluation.py \
    --model_name_or_path bert-base-uncased \
    --pooler hidden_states_subspace \
    --task_set sts \
    --mode test
