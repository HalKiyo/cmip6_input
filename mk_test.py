# 1873-2014 42models have responsible year->(4, 42, 24, 72), (42)
import numpy as np

WORKDIR = '/work/kajiyama/cnn/input'
SAVEDIR = '/work/kajiyama/cnn/test'
MODEL_NUM = 42
YEAR_NUM = 165


def main():
    indir = '/predictors/coarse_std'
    tors = 'predictors_coarse_std_Apr_o'
    outdir = '/pr/continuous/thailand/1x1'
    tant = 'pr_1x1_std_MJJASO_thailand'
    typ_flg = 'thailand'
    grid = 20
    test_selection(tors, tant, indir, outdir, typ_flg=typ_flg, grid=grid)


def load(inkey, outkey, indir, outdir):
    infile = WORKDIR + indir + f"/{inkey}.npy"
    outfile = WORKDIR + outdir + f"/{outkey}.npy"
    predictors = np.squeeze(np.load(infile))
    predictant = np.squeeze(np.load(outfile))
    return predictors, predictant


def test_mask():
    """Boolean (model, year) mask, True for the test sample of each model.

    Model i is tested on year index i + (YEAR_NUM - MODEL_NUM), that is one
    year from 1973 to 2014 per model. Boolean indexing walks the mask in the
    same row major order as the original nested loops.
    """
    mask = np.zeros((MODEL_NUM, YEAR_NUM), dtype=bool)
    mask[np.arange(MODEL_NUM), np.arange(MODEL_NUM) + (YEAR_NUM - MODEL_NUM)] = True
    return mask


def test_selection(inkey, outkey, indir, outdir, typ_flg='continuous', grid=20):
    save_predictors = f"{SAVEDIR}/val/val_{inkey}_1973-2014.npy"
    save_predictant = f"{SAVEDIR}/val/val_{outkey}_1973-2014.npy"
    train_predictors = f"{SAVEDIR}/train/train_{inkey}_1973-2014.npy"
    train_predictant = f"{SAVEDIR}/train/train_{outkey}_1973-2014.npy"

    predictors, predictant = load(inkey, outkey, indir, outdir)

    x_train = np.empty((MODEL_NUM*YEAR_NUM - MODEL_NUM, 24, 72))
    y_train = np.empty(YEAR_NUM*MODEL_NUM - MODEL_NUM)

    x_test = np.empty((MODEL_NUM, 24, 72))
    y_test = np.empty(MODEL_NUM)

    if typ_flg in ('one', 'thailand'):
        print(typ_flg)
        if typ_flg == 'thailand':
            y_train = np.empty((YEAR_NUM*MODEL_NUM - MODEL_NUM, grid, grid))
            y_test = np.empty((MODEL_NUM, grid, grid))
        mask = test_mask()
        # predictors (and predictant) may have more models or years than used here
        predictors = predictors[:MODEL_NUM, :YEAR_NUM]
        predictant = predictant[:MODEL_NUM, :YEAR_NUM]
        x_test[...] = predictors[mask]
        y_test[...] = predictant[mask]
        x_train[...] = predictors[~mask]
        y_train[...] = predictant[~mask]
        print(len(x_train))

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
