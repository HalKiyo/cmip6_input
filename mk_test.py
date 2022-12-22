# 1873-2014 42models have responsible year->(42, 24, 72, 5), (42)
import numpy as np

def load(inkey, outkey):
    infile = f"/docker/mnt/d/research/D2/resnet/predictors/{inkey}.npy"
    outfile = f"/docker/mnt/d/research/D2/resnet/predictant/{outkey}.npy"
    predictors = np.load(infile)
    predictant = np.load(outfile)
    return predictors, predictant

def make_test(inkey, outkey):
    savedir = f"/docker/mnt/d/research/D2/resnet/test"
    save_predictors = f"{savedir}/{inkey}_1973-2014.npy"
    save_predictant = f"{savedir}/{outkey}_173-2014.npy"
    predictors, predictant = load(inkey, outkey)
    x_test, y_test, = np.empty((42, 24, 72, predictors. shape[-1])), np.empty(42)
    for i in range(42):
         x_test[i, :, :, :] = predictors[0+i, 172+i]
         y_test[i] = predictant[0+i, 172+i]

