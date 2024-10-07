import pytest
from sorted_list import sorted_list

# Sample test for basic functionality
def test_sorted_list_basic():
    assert sorted_list([3, 1, 2, 4]) == [1, 2, 3, 4]
    assert sorted_list([10, 20, 30]) == [10, 20, 30]

# Testing for duplicate removal
def test_sorted_list_duplicates():
    assert sorted_list([1, 2, 2, 3, 4, 4]) == [1, 2, 3, 4]

# Edge case: empty input list
def test_sorted_list_empty():
    assert sorted_list([]) == []

# Edge case: input with all identical values
def test_sorted_list_identical_values():
    assert sorted_list([5, 5, 5, 5]) == [5]

# Negative test: input is not a list
def test_sorted_list_invalid_input():
    with pytest.raises(ValueError):
        sorted_list("not a list")
