import torch
from tensor_operations import add_tensors

def test_add_tensors():
    # Create two tensors
    tensor_a = torch.tensor([1.0, 2.0, 3.0])
    tensor_b = torch.tensor([4.0, 5.0, 6.0])

    # Add the tensors using the function
    result = add_tensors(tensor_a, tensor_b)

    # Expected result tensor
    expected = torch.tensor([5.0, 7.0, 9.0])

    # Check if result and expected tensor are the same using torch.allclose
    assert torch.allclose(result, expected), "Tensor addition result is incorrect"
