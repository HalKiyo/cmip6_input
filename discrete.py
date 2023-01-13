import bisect
import numpy as np
from os.path import exists

def EFD(data, class_num=5):
    mjjaso_one = data.copy() # data=(42, 165)
    one_flat = mjjaso_one.reshape(42*165)
    flat_sorted = np.sort(one_flat)
    if len(flat_sorted)%class_num != 0:
        print('class_num is wrong')
    else:
        batch_sample = int(len(flat_sorted)/class_num)

    bnd = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    bnd.append(flat_sorted[-1]+1e-10) # max boundary must be a bit higher than real max
    bnd = np.array(bnd)
    one_class = np.empty(len(one_flat))
    for i, value in enumerate(one_flat):
        label = bisect.bisect(bnd, value)
        one_class[i] = int(label - 1)
    one_class.reshape(42, 165)
    return one_class, bnd # one_class=(6930), bnd=(class_num+1)

def EWD(data, class_num=5):
    mjjaso_one = data.copy() # data=(42, 165)
    one_flat = mjjaso_one.reshape(42*165)
    lim = max(abs(max(one_flat)), abs(min(one_flat)))
    dx = 2*lim/class_num

    bnd = []
    bnd.append(-lim-1e-10) # min boundary must be a bit lower than real min
    bnd.append(lim+1e-10) # max boundary must be a bit higher than real max

    # even or odd
    if class_num%2 == 0:
        origin = 0
        bnd.append(origin)
    else:
        origin = dx/2
        bnd.append(origin)
        bnd.append(-origin)

    # EWD_bnd
    if class_num == 4 or class_num == 5:
        bnd.append(origin+dx)
        bnd.append(-origin-dx)
    elif class_num >= 6:
        loop_num = int(class_num/2)
        for i in range(loop_num-1):
            bnd.append(origin+dx*(i+1))
            bnd.append(-origin-dx*(i+1))
    bnd = np.sort(bnd)

    # EWD_conversion
    one_class = np.empty(len(one_flat))
    for i, value in enumerate(one_flat):
        label = bisect.bisect(bnd, value) # giving label
        one_class[i] = int(label - 1)
    bnd = np.array(bnd)

    u, counts = np.unique(one_class, return_counts=True)
    print(f"count: {counts}")
    print(f"bnd: {bnd}")
    print(f"max, min: {max(one_flat)}, {min(one_flat)}")
    return one_class, bnd # one_class=(6930), bnd=(class_num+1)
