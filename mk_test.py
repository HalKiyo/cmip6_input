# 1873-2014 42models have responsible year->(4, 42, 24, 72), (42)
import numpy as np

def main():
    indir = '/predictors/coarse_std'
    tors = 'predictors_coarse_std_Apr_o'
    outdir = '/pr/continuous/thailand/1x1'
    tant = 'pr_1x1_std_MJJASO_thailand'
    typ_flg = 'thailand'
    grid = 20
    test_selection(tors, tant, indir, outdir, typ_flg=typ_flg, grid=grid)

def load(inkey, outkey, indir, outdir):
    workdir = '/work/kajiyama/cnn/input'
    infile = workdir + indir + f"/{inkey}.npy"
    outfile = workdir + outdir + f"/{outkey}.npy"
    predictors = np.squeeze(np.load(infile))
    predictant = np.squeeze(np.load(outfile))
    return predictors, predictant

def test_selection(inkey, outkey, indir, outdir, typ_flg='continuous', grid=20):
    savedir = f"/work/kajiyama/cnn/test"
    save_predictors = f"{savedir}/val/val_{inkey}_1973-2014.npy"
    save_predictant = f"{savedir}/val/val_{outkey}_1973-2014.npy"
    train_predictors = f"{savedir}/train/train_{inkey}_1973-2014.npy"
    train_predictant = f"{savedir}/train/train_{outkey}_1973-2014.npy"

    predictors, predictant = load(inkey, outkey, indir, outdir)

    x_train = np.empty((42*165 - 42, 24, 72))
    y_train = np.empty(165*42 - 42)

    x_test = np.empty((42, 24, 72))
    y_test = np.empty(42)

    if typ_flg == 'one':
        print('one')
        index = 0
        for i in range(42):
            for j in range(165):
                if i == (j-(165-42)):
                    x_test[i, :, :] = predictors[i, j]
                    y_test[i] = predictant[i, j]
                else:
                    x_train[index, :, :] = predictors[i, j]
                    y_train[index] = predictant[i, j]
                    index += 1
                    print(index)

    elif typ_flg == 'thailand':
        print('thailand')
        index = 0
        y_train = np.empty((165*42 - 42, grid, grid))
        y_test = np.empty((42, grid, grid))
        for i in range(42):
            for j in range(165):
                if i == (j-(165-42)):
                    x_test[i, :, :] = predictors[i, j]
                    y_test[i, :, :] = predictant[i, j, :, :]
                else:
                    x_train[index, :, :] = predictors[i, j]
                    y_train[index, :, :] = predictant[i, j, :, :]
                    index += 1
                    print(index)

    np.save(train_predictors, np.squeeze(x_train))
    np.save(train_predictant, y_train)

    np.save(save_predictors, np.squeeze(x_test))
    np.save(save_predictant, y_test)

def test_random():
    pass

if __name__ == '__main__':
    try:
        main()
        print('[\N{check mark}]')
    except Exception as e:
        print('[\N{cross mark}]')
        print(e)

