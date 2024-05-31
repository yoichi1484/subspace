#!/bin/bash

# SubspaceBERTScore F
python evaluation.py \
    --model_name_or_path bert-base-uncased \
    --pooler hidden_states_subspace_bert_score_F \
    --task_set sts \
    --mode test

# SubspaceBERTScore P
#python evaluation.py \
#    --model_name_or_path bert-base-uncased \
#    --pooler hidden_states_subspace_bert_score_P \
#    --task_set sts \
#    --mode test

# SubspaceBERTScore R
#python evaluation.py \
#    --model_name_or_path bert-base-uncased \
#    --pooler hidden_states_subspace_bert_score_R \
#    --task_set sts \
#    --mode test
