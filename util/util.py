def get_tuple(x, dim=2):
    """
    Take in a int or tuple and return a tuple of ints, one for each dimension

    Args:
        x (int or tuple[int]): The input to (possibly) convert to a tuple of ints
        dim (int): The number of elements for the return tuple
        
    Return:
        A tuple of ints
    """
    if isinstance(x, int):
        return tuple(
            [x for _ in range(dim)]
        )
    return x