# Subspace-based Set Operations on a Pre-trained Word Embedding Space



## Set similarity
Sentence similarity can be calculated using our set similarity.

```python
from subspace.tool import SubspaceJohnsonSimilarity

scorer = SubspaceJohnsonSimilarity(device='cpu', model_name_or_path='bert-base-uncased')

sentences_a = ["A man with a hard hat is dancing.", "A young child is riding a horse."]
sentences_b = ["A man wearing a hard hat is dancing.", "A child is riding a horse."]

scorer(sentences_a, sentences_b) 
```

Evaluation experiments on the STS task will be conducted with ```SentEval```. 
The evaluation scripts and the calculation of correlation coefficients are based on the code of (Gao & Yao)[https://github.com/princeton-nlp/SimCSE].
```
$ sh run_sts.sh
```


## Other set operations
The basis of union, intersection, orthogonal complement, etc. based on quantum logic can be computed as follows.

```python
import numpy as np
from subspace.operations import *

np.random.seed(0)
A = np.random.random_sample((50, 300))
B = np.random.random_sample((80, 300))
```

Compute bases of the subspace
```python
SA = subspace(A)
SA.shape # (90, 300)
```

Compute bases of the orthogonal complement
```python
A_NOT = orthogonal_complement(A)
A_NOT.shape # (210, 300)
```

Compute bases of the intersection
```python
A_AND_B = intersection(A, B)
A_AND_B.shape # (1, 300)
```

Compute bases of the sum space
```python
A_OR_B = sum_space(A, B)
A_OR_B.shape # (180, 300)
```

Compute soft membership degree
```python
v = np.random.random_sample(300,) 
soft_membership(A, v) # 0.89
```