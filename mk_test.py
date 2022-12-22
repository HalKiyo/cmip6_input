# 1873-2014 42models have responsible year->(4, 42, 24, 72), (42)
import numpy as np

def main():
    tors = 'predictors_coarse_std_Apr_msot'
    tant = 'pr_1x1_std_MJJASO_one'
    make_test(tors, tant)

def load(inkey, outkey):
    infile = f"/work/kajiyama/cnn/input/predictors/coarse_std/{inkey}.npy"
    outfile = f"/work/kajiyama/cnn/input/pr/one/1x1/{outkey}.npy"
    predictors = np.load(infile)
    predictant = np.load(outfile)
    return predictors, predictant

def make_test(inkey, outkey):
    savedir = f"/work/kajiyama/cnn/test"
    save_predictors = f"{savedir}/{inkey}_1973-2014.npy"
    save_predictant = f"{savedir}/{outkey}_1973-2014.npy"
    predictors, predictant = load(inkey, outkey)
    x_test, y_test, = np.empty((predictors.shape[0], 42, 24, 72)), np.empty(42)
    for i in range(42):
        x_test[:, i, :, :] = predictors[:, 0+i, (165-42)+i]
        y_test[i] = predictant[0+i, (165-42)+i]
    np.save(save_predictors, x_test)
    np.save(save_predictant, y_test)

if __name__ == '__main__':
    try:
        main()
        print('[\N{check mark}]')
    except Exception as e:
        print('[\N{cross mark}]')
        print(e)

