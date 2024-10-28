## ==================================================================
## File    : test_util_convetranspose2d_spape.py
## Purpose : Test the utility function convetranspose2d_spape.py
## Authon  : spandit
## ==================================================================

import pytest
from yoke.models.cnn_utils import convtranspose2d_shape

# Assuming the function convtranspose2d_shape is imported from the relevant module
# from module_name import convtranspose2d_shape

def test_convtranspose2d_shape():
    # Test 1: Test with basic values
    w, h       = 4, 4
    k_w, k_h   = 3, 3
    s_w, s_h   = 2, 2
    p_w, p_h   = 1, 1
    op_w, op_h = 0, 0
    d_w, d_h   = 1, 1

    expected_w = (w - 1) * s_w - 2 * p_w + d_w *(k_w - 1) + op_w + 1
    expected_h = (h - 1) * s_h - 2 * p_h + d_h *(k_h - 1) + op_h + 1
    expected_total = expected_w * expected_h

    new_w, new_h, total = convtranspose2d_shape(w,    h,
                                                k_w,  k_h,
                                                s_w,  s_h,
                                                p_w,  p_h,
                                                op_w, op_h,
                                                d_w,  d_h)

    assert new_w == expected_w, f"Expected width {expected_w}, but got {new_w}"
    assert new_h == expected_h, f"Expected height {expected_h}, but got {new_h}"
    assert total == expected_total, f"Expected total {expected_total}, but got {total}"

    # Test 2: Test with output padding
    w, h       = 4, 4
    k_w, k_h   = 3, 3
    s_w, s_h   = 2, 2
    p_w, p_h   = 1, 1
    op_w, op_h = 1, 1
    d_w, d_h   = 1, 1

    expected_w = (w - 1) * s_w - 2 * p_w + d_w *(k_w - 1) + op_w + 1
    expected_h = (h - 1) * s_h - 2 * p_h + d_h *(k_h - 1) + op_h + 1
    expected_total = expected_w * expected_h

    new_w, new_h, total = convtranspose2d_shape(w,    h,
                                                k_w,  k_h,
                                                s_w,  s_h,
                                                p_w,  p_h,
                                                op_w, op_h,
                                                d_w,  d_h)

    assert new_w == expected_w, f"Expected width {expected_w}, but got {new_w}"
    assert new_h == expected_h, f"Expected height {expected_h}, but got {new_h}"
    assert total == expected_total, f"Expected total {expected_total}, but got {total}"

