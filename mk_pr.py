# example = 'pr_5x5_std_MJJ_thailand_ ***|*** mrso_snc_tos_tsl.npy'
# boxplot: x = year, y= diviation of monthly rainfall
# boxplot: close up only 1850, 1900, 1950, 2000

import sys

import numpy as np

WORKDIR = '/work/kajiyama/cnn/input/pr'
VARIABLE = 'pr'
LEAD_LST = [1, 3, 6]
SPACE_LST = ['monsoon']
#SPACE_LST = ['one', 'thailand', 'world']
MODEL_NUM = 42
YEAR_NUM = 165

# Region bounds in 5x5 grid units. They are multiplied by the number of
# 1x1 cells per 5x5 cell (5) for the 1x1 data.
LAT_LST = [7, 11]  # 5-25N
LON_LST = [18, 22]  # 90-110E
MA_LAT = [1, 15]  # 15S-55N
MA_LON = [12, 30]  # 60-150E

# resolution: (grid scale, input forms)
RES_CONFIG = {
    '1x1': (5, ['raw', 'anom', 'std']),
    '5x5': (1, ['coarse', 'coarse_anom', 'coarse_std']),
}


def main():
    pr_5x5()
    pr_1x1()


def lead_cnd(leadtime):
    if leadtime == 1:
        mon_arr = np.array([[4,5], [5,6], [6,7], [7,8], [8,9], [9,10]])
        llst = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    elif leadtime == 3:
        mon_arr = np.array([[4,7], [5,8], [6,9], [7,10]])
        llst = ['MJJ', 'JJA', 'JAS', 'ASO']
    elif leadtime == 6:
        mon_arr = np.array([[4,10]])
        llst = ['MJJASO']
    else:
        print('error of lead_cnd')
        sys.exit()
    return mon_arr, llst


def make_pr(resolution):
    scale, form_lst = RES_CONFIG[resolution]
    lat_lst = [v*scale for v in LAT_LST]
    lon_lst = [v*scale for v in LON_LST]
    ma_lat = [v*scale for v in MA_LAT]
    ma_lon = [v*scale for v in MA_LON]
    msk = 4*scale  # number of grid cells per side of the thailand box

    for form in form_lst:
        ifile = f"{WORKDIR}/main/{VARIABLE}_{form}.npy"
        loaded = np.load(ifile)
        print(f"{ifile} is loaded")

        for leadtime in LEAD_LST:
            mon_arr, llst = lead_cnd(leadtime)
            for i, months in enumerate(llst):
                m0, m1 = mon_arr[i, 0], mon_arr[i, 1]
                for space in SPACE_LST:

                    if space == 'one':
                        one = loaded[:, :, m0:m1, lat_lst[0]:lat_lst[1], lon_lst[0]:lon_lst[1]]
                        out = np.mean(one.reshape(MODEL_NUM, YEAR_NUM, leadtime*msk*msk), axis=2)

                    elif space == 'thailand':
                        thailand = loaded[:, :, m0:m1, lat_lst[0]:lat_lst[1], lon_lst[0]:lon_lst[1]]
                        out = np.mean(thailand, axis=2)

                    elif space == 'monsoon':
                        monsoon = loaded[:, :, m0:m1, ma_lat[0]:ma_lat[1], ma_lon[0]:ma_lon[1]]
                        out = np.mean(monsoon, axis=2)

                    elif space == 'world':
                        world = loaded[:, :, m0:m1, :, :]
                        out = np.mean(world, axis=2)

                    ofile = f"{WORKDIR}/{VARIABLE}_{resolution}_{form}_{months}_{space}.npy"
                    np.save(ofile, out)
                    print(f"{ofile} is saved {out.shape}")


def pr_1x1():
    make_pr('1x1')


def pr_5x5():
    make_pr('5x5')


if __name__ == '__main__':
    main()
