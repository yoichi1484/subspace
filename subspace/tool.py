import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from numpy import ndarray
import numpy as np
from .similarity import subspace_johnson, subspace_bert_score, vanilla_bert_score


class MySimilarity:
    def __init__(self, device='cpu', model_name_or_path='bert-base-uncased'):
        # Set up model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(device)
        self.max_length = 128

    def __call__(self, sentence1, sentence2, weight="L2"):
        pass
    
    
    def encode(self, sentence, return_numpy=False, batch_size=12):
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in range(total_batch):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)

                embeddings = outputs.last_hidden_state.cpu()
                embedding_list.append(embeddings)
                
        embeddings = torch.cat(embedding_list, 0)
        
        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings


class SubspaceJohnsonSimilarity(MySimilarity):
    def __call__(self, sentence1, sentence2, weight="L2"):
        hidden_states1 = self.encode(sentence1)
        hidden_states2 = self.encode(sentence2)
        return subspace_johnson(hidden_states1, hidden_states2, weight)
    
        
class SubspaceBERTScore(MySimilarity):
    def __call__(self, sentence1, sentence2, weight="L2"):
        hidden_states1 = self.encode(sentence1)
        hidden_states2 = self.encode(sentence2)
        return subspace_bert_score(hidden_states1, hidden_states2, weight)
    
    
class VanillaBERTScore(MySimilarity):
    def __call__(self, sentence1, sentence2, weight="L2"):
        hidden_states1 = self.encode(sentence1)
        hidden_states2 = self.encode(sentence2)
        return vanilla_bert_score(hidden_states1, hidden_states2, weight)