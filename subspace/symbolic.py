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
