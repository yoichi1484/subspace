# Subspace-based Set Operations on a Pre-trained Word Embedding Space
Yoichi Ishibashi, Sho Yokoi, Katsuhito Sudoh, Satoshi Nakamura: [Subspace-based Set Operations on a Pre-trained Word Embedding Space](https://arxiv.org/abs/2210.13034) (Preprint, 2022)


## Setup
Install the required packages.
```
pip install -r requirements.txt
```

## Set similarity
Our subspace-based sentence (set of words) similarity can be easily computed as follows.

### Usage
```python
from subspace.tool import SubspaceJohnsonSimilarity

scorer = SubspaceJohnsonSimilarity(device='cpu', model_name_or_path='princeton-nlp/unsup-simcse-bert-base-uncased')

sentences_a = ["A man with a hard hat is dancing.", "A young child is riding a horse."]
sentences_b = ["A man wearing a hard hat is dancing.", "A child is riding a horse."]

scorer(sentences_a, sentences_b) # tensor([1.9746, 1.9562])
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
bash run_sts.sh
```


## Other set operations
Other subspace-based set operations such as union, intersection, orthogonal complement, and soft membership can be computed as follows.

```python
import numpy as np
from subspace.operations import *

np.random.seed(0)
A = np.random.random_sample((50, 300)) # 50 stacked 300-dimensional word vectors
B = np.random.random_sample((80, 300)) # 80 stacked 300-dimensional word vectors
```

Compute bases of the subspace
```python
SA = subspace(A)
SA.shape # (50, 300)
```

Compute bases of the orthogonal complement
```python
A_NOT = orthogonal_complement(A)
A_NOT.shape # (250, 300)
```

Compute bases of the intersection
```python
A_AND_B = intersection(A, B)
A_AND_B.shape # (1, 300)
```

Compute bases of the sum space
```python
A_OR_B = sum_space(A, B)
A_OR_B.shape # (130, 300)
```

Compute soft membership degree
```python
v = np.random.random_sample(300,) 
soft_membership(A, v) # 0.89
```

## Citation
```bibtex
@inproceedings{Ishibashi:Subspace:2022,
  author = {Yoichi Ishibashi, Sho Yokoi, Katsuhito Sudoh, Satoshi Nakamura},  
  title = {Subspace-based Set Operations on a Pre-trained Word Embedding Space},
  year = {2022}
}
```
