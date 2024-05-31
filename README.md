# Subspace Representations for Soft Set Operations and Sentence Similarities
Yoichi Ishibashi, Sho Yokoi, Katsuhito Sudoh, Satoshi Nakamura: [Subspace Representations for Soft Set Operations and Sentence Similarities](https://arxiv.org/abs/2210.13034) (NAACL, 2024)


## Setup
Install the required packages.
```
cd subspace
pip install -r requirements.txt
```

## Set similarity
Our subspace-based sentence (set of words) similarity can be easily computed as follows.

### Usage
```python
from subspace.tool import SubspaceBERTScore

scorer = SubspaceBERTScore(device='cpu', model_name_or_path='bert-base-uncased')

sentences_a = ["A man with a hard hat is dancing.", "A young child is riding a horse."]
sentences_b = ["A man wearing a hard hat is dancing.", "A child is riding a horse."]

scorer(sentences_a, sentences_b)
```

### STS task
Evaluation experiments on the STS task can be conducted with ```SentEval```. 
The first step is to download the evaluation data.
```
cd SentEval/data/downstream/
bash download_dataset.sh
```

The evaluation scripts and the calculation of correlation coefficients are based on the code of [Gao & Yao](https://github.com/princeton-nlp/SimCSE).
Here is how to run the script:
```
cd ../../../
bash run_sts.sh
```

## Other set operations
Other subspace-based set operations such as union, intersection, orthogonal complement, and soft membership can be computed as follows using torch.

```python
import torch
from subspace.operations import *

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A = torch.rand((50, 300), device=device) # 50 stacked 300-dimensional word vectors
B = torch.rand((80, 300), device=device) # 80 stacked 300-dimensional word vectors
```

Compute bases of the subspace
```python
SA = subspace(A)
SA.shape # torch.Size([50, 300])
```

Compute bases of the orthogonal complement
```python
A_NOT = orthogonal_complement(A)
A_NOT.shape # torch.Size([250, 300])
```

Compute bases of the intersection
```python
A_AND_B = intersection(A, B)
A_AND_B.shape # torch.Size([1, 300])
```

Compute bases of the sum space
```python
A_OR_B = sum_space(A, B)
A_OR_B.shape # torch.Size([130, 300])
```

Compute soft membership degree
```python
v = torch.rand(300, device=device)
soft_membership(A, v) # tensor(0.89)
```

### Note
The previous numpy-based operations have been moved to a separate folder. If you still need to use them, you can find them in the `subspace/legacy_operations` folder. 
