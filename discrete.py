"""Shared discretization helpers.

EFD splits the data into classes with an equal number of samples.
EWD splits the data into classes of equal width that are symmetric around zero.
mk_class.py builds on the helpers below.
"""
import bisect

import numpy as np

EPS = 1e-10  # margin so that the real min and max fall inside the outer boundaries


def classify(data, bnd):
    """Label every element with bisect.bisect(bnd, value) - 1.

    The result is a float64 array with the same shape as data, exactly as the
    original per element loops produced. np.searchsorted with side='right' gives
    the same answer as bisect.bisect as long as bnd has no NaN, so the original
    loop is kept as a fallback for that case.
    """
    data = np.asarray(data)
    out = np.empty(data.shape)
    if np.isnan(bnd).any():
        flat = out.reshape(-1)
        for i, value in enumerate(data.reshape(-1)):
            flat[i] = int(bisect.bisect(bnd, value) - 1)
    else:
        out[...] = np.searchsorted(bnd, data, side='right') - 1
    return out


def efd_bnd(flat, class_num=5, lower_margin=True):
    """Equal frequency boundaries (class_num + 1 values) of a 1D array."""
    flat_sorted = np.sort(flat)
    if len(flat_sorted) % class_num != 0:
        print('class_num is wrong')
    else:
        batch_sample = int(len(flat_sorted) / class_num)

    bnd = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    bnd.append(flat_sorted[-1] + EPS)  # max boundary must be a bit higher than real max
    if lower_margin:
        bnd[0] = bnd[0] - EPS  # min boundary must be a bit lower than real min
    return np.array(bnd)


def ewd_bnd(flat, class_num=5):
    """Equal width boundaries of a 1D array, symmetric around zero."""
    lim = max(abs(max(flat)), abs(min(flat)))
    dx = 2 * lim / class_num

    bnd = []
    bnd.append(-lim - EPS)  # min boundary must be a bit lower than real min
    bnd.append(lim + EPS)  # max boundary must be a bit higher than real max

    # even or odd
    if class_num % 2 == 0:
        origin = 0
        bnd.append(origin)
    else:
        origin = dx / 2
        bnd.append(origin)
        bnd.append(-origin)

    if class_num == 4 or class_num == 5:
        bnd.append(origin + dx)
        bnd.append(-origin - dx)
    elif class_num >= 6:
        loop_num = int(class_num / 2)
        for i in range(loop_num - 1):
            bnd.append(origin + dx * (i + 1))
            bnd.append(-origin - dx * (i + 1))
    return np.sort(bnd)


def ewd_report(classes, bnd, flat, show_labels=True):
    """Print the class counts and boundaries of an EWD result."""
    u, counts = np.unique(classes, return_counts=True)
    if show_labels:
        print(f"class_label: {u}")
    print(f"count: {counts}")
    print(f"bnd: {bnd}")
    print(f"max, min: {max(flat)}, {min(flat)}")


def EFD(data, class_num=5):
    one_flat = data.reshape(42 * 165)  # data=(42, 165)
    bnd = efd_bnd(one_flat, class_num=class_num, lower_margin=False)
    one_class = classify(one_flat, bnd)
    return one_class, bnd  # one_class=(6930), bnd=(class_num+1)


def EWD(data, class_num=5):
    one_flat = data.reshape(42 * 165)  # data=(42, 165)
    bnd = ewd_bnd(one_flat, class_num=class_num)
    one_class = classify(one_flat, bnd)
    ewd_report(one_class, bnd, one_flat, show_labels=False)
    return one_class, bnd  # one_class=(6930), bnd=(class_num+1)
