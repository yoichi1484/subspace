# Copyright 2018 Babylon Partners. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

def symbolic_johnson(x, y):
    """
    Classical Johnson similarity measure between two sets
    :param x: list of words (strings) for the first sentence
    :param y: list of words (strings) for the second sentence
    :return: similarity score between two sentences
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0
    xs = set(x)
    ys = set(y)
    inter = xs & ys
    return len(inter) / len(xs) + len(inter) / len(ys)


def symbolic_jaccard(x, y):
    """
    Classical Jaccard similarity measure between two sets
    :param x: list of words (strings) for the first sentence
    :param y: list of words (strings) for the second sentence
    :return: similarity score between two sentences
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0
    xs = set(x)
    ys = set(y)
    inter = xs & ys
    union = xs | ys
    return len(inter) / len(union)
