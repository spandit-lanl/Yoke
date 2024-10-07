def sorted_list(list_numbers):
    if not isinstance(list_numbers, list):
        raise ValueError("Input should be a list")
    return sorted(set(list_numbers))
