import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Import similarities
import similarity
#from subspace.similarity import subspace_johnson
import subspace

def subspace_bert_score_F(x, y, weight = "L2"):
    P, R, F = subspace.subspace_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return F.numpy()[0]

def subspace_bert_score_P(x, y, weight = "L2"):
    P, R, F = subspace.subspace_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return P.numpy()[0]

def subspace_bert_score_R(x, y, weight = "L2"):
    P, R, F = subspace.subspace_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return R.numpy()[0]

def subspace_bert_score_F_noweight(x, y, weight = "no"):
    P, R, F = subspace.subspace_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return F.numpy()[0]

def subspace_bert_score_P_noweight(x, y, weight = "no"):
    P, R, F = subspace.subspace_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return P.numpy()[0]

def subspace_bert_score_R_noweight(x, y, weight = "no"):
    P, R, F = subspace.subspace_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return R.numpy()[0]

def vanilla_bert_score_F(x, y, weight = "L2"):
    P, R, F = subspace.vanilla_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return F.numpy()[0]

def vanilla_bert_score_P(x, y, weight = "L2"):
    P, R, F = subspace.vanilla_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return P.numpy()[0]

def vanilla_bert_score_R(x, y, weight = "L2"):
    P, R, F = subspace.vanilla_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return R.numpy()[0]

def vanilla_bert_score_F_noweight(x, y, weight = "no"):
    P, R, F = subspace.vanilla_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return F.numpy()[0]

def vanilla_bert_score_P_noweight(x, y, weight = "no"):
    P, R, F = subspace.vanilla_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return P.numpy()[0]

def vanilla_bert_score_R_noweight(x, y, weight = "no"):
    P, R, F = subspace.vanilla_bert_score(x.unsqueeze(0), y.unsqueeze(0), weight)
    return R.numpy()[0]

def subspace_johnson(x, y, weight = "L2"):
    return subspace.subspace_johnson(x.unsqueeze(0), y.unsqueeze(0), weight).numpy()[0]

def dynamax_jaccard(x, y):
    return similarity.dynamax_jaccard(x.numpy(), y.numpy())

def symbolic_johnson(x, y):
    return similarity.symbolic_johnson(x, y)

def symbolic_jaccard(x, y):
    return similarity.symbolic_jaccard(x, y)

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 
                     'cls_before_pooler',                 # CLS-cos
                     'avg', 'avg_top2', 'avg_first_last', # Avg-cos
                     'hidden_states_subspace_johnson',      # SubspaceJohnson
                     'hidden_states_subspace_bert_score_F', # SubspaceBERTScore
                     'hidden_states_subspace_bert_score_P', # SubspaceBERTScore
                     'hidden_states_subspace_bert_score_R', # SubspaceBERTScore
                     'hidden_states_subspace_bert_score_F_noweight', # SubspaceBERTScore
                     'hidden_states_subspace_bert_score_P_noweight', # SubspaceBERTScore
                     'hidden_states_subspace_bert_score_R_noweight', # SubspaceBERTScore
                     'hidden_states_vanilla_bert_score_F',  # BERTScore
                     'hidden_states_vanilla_bert_score_P',  # BERTScore
                     'hidden_states_vanilla_bert_score_R',  # BERTScore
                     'hidden_states_vanilla_bert_score_F_noweight',  # BERTScore
                     'hidden_states_vanilla_bert_score_P_noweight',  # BERTScore
                     'hidden_states_vanilla_bert_score_R_noweight',  # BERTScore
                     'hidden_states_dynamax',             # DynaMax
                     "words_for_symbolic_johnson",        # Symbolic set similarity (Johnson)
                     "words_for_symbolic_jaccard"],       # Symbolic set similarity (Jaccard)
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, \
                        dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            #choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     #'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    
    args = parser.parse_args()
    
    print("Pooler and similarity: ", args.pooler)
    
    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    device = torch.device("cpu")
    model = model.to(device)
    
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    #elif args.task_set == 'transfer':
    #    args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    #elif args.task_set == 'full':
    #    args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    #    args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError
        
    params['pooler'] = args.pooler 
    params['model_name'] = args.model_name_or_path
    
    if args.pooler in ["hidden_states_subspace_johnson", "hidden_states_first_last_subspace_johnson"]: 
        params['similarity'] = subspace_johnson  
        
    elif args.pooler in ["hidden_states_dynamax", "hidden_states_first_last_dynamax"]:
        params['similarity'] = dynamax_jaccard
        
    elif args.pooler in ["hidden_states_subspace_bert_score_F", "hidden_states_first_last_subspace_bert_score_F"]: 
        params['similarity'] = subspace_bert_score_F
        
    elif args.pooler in ["hidden_states_subspace_bert_score_P", "hidden_states_first_last_subspace_bert_score_P"]: 
        params['similarity'] = subspace_bert_score_P
        
    elif args.pooler in ["hidden_states_subspace_bert_score_R", "hidden_states_first_last_subspace_bert_score_R"]: 
        params['similarity'] = subspace_bert_score_R
        
    elif args.pooler in ["hidden_states_subspace_bert_score_F_noweight"]: 
        params['similarity'] = subspace_bert_score_F_noweight
        
    elif args.pooler in ["hidden_states_subspace_bert_score_P_noweight"]: 
        params['similarity'] = subspace_bert_score_P_noweight
        
    elif args.pooler in ["hidden_states_subspace_bert_score_R_noweight"]: 
        params['similarity'] = subspace_bert_score_R_noweight
        
    elif args.pooler in ["hidden_states_vanilla_bert_score_F", "hidden_states_first_last_vanilla_bert_score_F"]: 
        params['similarity'] = vanilla_bert_score_F
        
    elif args.pooler in ["hidden_states_vanilla_bert_score_P", "hidden_states_first_last_vanilla_bert_score_P"]: 
        params['similarity'] = vanilla_bert_score_P
        
    elif args.pooler in ["hidden_states_vanilla_bert_score_R", "hidden_states_first_last_vanilla_bert_score_R"]: 
        params['similarity'] = vanilla_bert_score_R
        
    elif args.pooler in ["hidden_states_vanilla_bert_score_F_noweight"]: 
        params['similarity'] = vanilla_bert_score_F_noweight
        
    elif args.pooler in ["hidden_states_vanilla_bert_score_P_noweight"]: 
        params['similarity'] = vanilla_bert_score_P_noweight
        
    elif args.pooler in ["hidden_states_vanilla_bert_score_R_noweight"]: 
        params['similarity'] = vanilla_bert_score_R_noweight
        
    #elif args.pooler in ["hidden_states_wrd", "hidden_states_first_last_wrd"]:
    #    params['similarity'] = wrd
        
    #elif args.pooler in ["hidden_states_wmd", "hidden_states_first_last_wmd"]:
    #    params['similarity'] = wmd
        
    #elif args.pooler in ["hidden_states_grassmann", "hidden_states_first_last_grassmann"]:
    #    params['similarity'] = grassmann_distance
        
    elif args.pooler == "words_for_symbolic_johnson":
        params['similarity'] = symbolic_johnson 
        
    elif args.pooler == "words_for_symbolic_jaccard":
        params['similarity'] = symbolic_jaccard


    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        
        # Return words for symbolic set similarity
        if params['pooler'] in ['words_for_symbolic_johnson', 'words_for_symbolic_jaccard']:
            return batch

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        # Apply different poolers
        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        
        elif args.pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()

        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
            
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        
        elif "hidden_states_first_last" in args.pooler:
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            return first_hidden + last_hidden
        
        elif "hidden_states" in args.pooler:
            return last_hidden.cpu()
            
        else:
            raise NotImplementedError

    results = {}
            
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        #task_names = []
        #scores = []
        #for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        #    task_names.append(task)
        #    if task in results:
        #        scores.append("%.2f" % (results[task]['acc']))    
        #    else:
        #        scores.append("0.00")
        #task_names.append("Avg.")
        #scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        #print_table(task_names, scores)


if __name__ == "__main__":
    main()
