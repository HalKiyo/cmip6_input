# 1873-2014 42models have responsible year->(4, 42, 24, 72), (42)
import numpy as np

def main():
    indir = '/predictors/coarse_std'
    tors = 'predictors_coarse_std_Apr_o'
    outdir = '/pr/class/thailand/EFD'
    tant = 'pr_1x1_std_MJJASO_thailand_EFD_5'
    typ_flg = 'thailand'
    grid = 20
    make_test(tors, tant, indir, outdir, typ_flg=typ_flg, grid=grid)

def load(inkey, outkey, indir, outdir):
    workdir = '/work/kajiyama/cnn/input'
    infile = workdir + indir + f"/{inkey}.npy"
    outfile = workdir + outdir + f"/{outkey}.npy"
    predictors = np.squeeze(np.load(infile))
    predictant = np.squeeze(np.load(outfile))
    return predictors, predictant

def make_test(inkey, outkey, indir, outdir, typ_flg='continuous', grid=20):
    savedir = f"/work/kajiyama/cnn/test"
    save_predictors = f"{savedir}/{inkey}_1973-2014.npy"
    save_predictant = f"{savedir}/{outkey}_1973-2014.npy"
    predictors, predictant = load(inkey, outkey, indir, outdir)
    x_test = np.empty((42, 24, 72))
    y_test = np.empty(42)
    if typ_flg == 'one':
        print('one')
        for i in range(42):
            x_test[i, :, :] = predictors[0+i, (165-42)+i]
            y_test[i] = predictant[0+i, (165-42)+i]
    elif typ_flg == 'thailand':
        print('thailand')
        y_test = np.empty((42, grid, grid))
        for i in range(42):
            x_test[i, :, :] = predictors[0+i, (165-42)+i]
            print(predictors[0+i, (165-42)+i])
            exit()
            y_test[i, :, :] = predictant[0+i, (165-42)+i, :, :]
    np.save(save_predictors, np.squeeze(x_test))
    np.save(save_predictant, y_test)

def make_train():
    pass

if __name__ == '__main__':
    try:
        main()
        print('[\N{check mark}]')
    except Exception as e:
        print('[\N{cross mark}]')
        print(e)

