import pytest
from yoke.models.cnn_utils import conv2d_shape

def test_conv2d_shape():        
    # Define test cases: (w, h, k, s_w, s_h, p_w, p_h, expected_new_w, expected_new_h, expected_total)
    test_cases = [
            (100, 150, 3, 2, 1, 2, 1, 51, 150, 7650),
            (28, 28, 3, 1, 1, 1, 1, 28, 28, 784),
            (32, 32, 5, 1, 1, 0, 0, 28, 28, 784),
            (64, 64, 3, 2, 2, 1, 1, 32, 32, 1024)
            ]

    for w, h, k, s_w, s_h, p_w, p_h, expected_new_w, expected_new_h, expected_total in test_cases:    
        new_w, new_h, total = conv2d_shape(w, h, k, s_w, s_h, p_w, p_h)
        assert new_w == expected_new_w, f"Expected width {expected_new_w}, but got {new_w}"
        assert new_h == expected_new_h, f"Expected height {expected_new_h}, but got {new_h}"
        assert total == expected_total, f"Expected total {expected_total}, but got {total}"

