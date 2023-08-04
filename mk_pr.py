# example = 'pr_5x5_std_MJJ_thailand_ ***|*** mrso_snc_tos_tsl.npy' 
# boxplot: x = year, y= diviation of monthly rainfall
# boxplot: close up only 1850, 1900, 1950, 2000

import numpy as np

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
        exit()
    return mon_arr, llst

def space_cnd(leadtime, inp):
    if leadtime == 1:
        out = np.mean(inp, axis=2)
    elif leadtime == 3 or leadtime==6:
        out = np.mean(inp, axis=2)
    else:
        print('error of space_cnd')
        exit()
    return out

def pr_1x1():
    workdir = '/work/kajiyama/cnn/input/pr'

    resolution = '1x1'
    lat_lst = [7*5, 11*5] # 5-25N
    lon_lst = [18*5, 22*5] # 90-110E
    ma_lat = [1*5, 15*5] # 15S-55N
    ma_lon = [12*5, 30*5] # 60-150E
    form_lst = ['raw', 'anom', 'std']
    msk = 20

    variable = 'pr'
    lead_lst = [1, 3, 6]
    space_lst = ['monsoon']
    #space_lst = ['one', 'thailand', 'world']
    model_num = 42
    year_num = 165

    for form in form_lst:
        ifile = f"{workdir}/main/{variable}_{form}.npy"
        loaded = np.load(ifile)
        print(f"{ifile} is loaded")

        for leadtime in lead_lst:
            mon_arr, llst = lead_cnd(leadtime)
            for i, months in enumerate(llst):
                for space in space_lst:

                    if space == 'one':
                        one = loaded[:, :, mon_arr[i,0]:mon_arr[i,1], lat_lst[0]:lat_lst[1], lon_lst[0]:lon_lst[1]]
                        out = np.mean(one.reshape(model_num, year_num, leadtime*msk*msk), axis=2)

                    elif space == 'thailand':
                        thailand = loaded[:, :, mon_arr[i,0]:mon_arr[i,1], lat_lst[0]:lat_lst[1], lon_lst[0]:lon_lst[1]]
                        out = np.mean(thailand, axis=2)

                    elif space == 'monsoon':
                        monsoon = loaded[:, :, mon_arr[i,0]:mon_arr[i,1],  ma_lat[0]:ma_lat[1], ma_lon[0]:ma_lon[:]]
                        out = np.mean(monsoon, axis=2)

                    elif space == 'world':
                        world = loaded[:, :, mon_arr[i,0]:mon_arr[i,1], :, :]
                        out = np.mean(world, axis=2)

                    ofile = f"{workdir}/{variable}_{resolution}_{form}_{months}_{space}.npy"
                    np.save(ofile, out)
                    print(f"{ofile} is saved {out.shape}")

def pr_5x5():
    workdir = '/work/kajiyama/cnn/input/pr'

    resolution = '5x5'
    lat_lst = [7, 11] # 5-25N
    lon_lst = [18, 22] # 90-110E
    ma_lat = [1, 15] # 15S-55N
    ma_lon = [12, 30] # 60-150E
    form_lst = ['coarse', 'coarse_anom', 'coarse_std']
    msk = 4

    variable = 'pr'
    lead_lst = [1, 3, 6]
    space_lst = ['monsoon']
    #space_lst = ['one', 'thailand', 'world']
    model_num = 42
    year_num = 165

    for form in form_lst:
        ifile = f"{workdir}/main/{variable}_{form}.npy"
        loaded = np.load(ifile)

        for leadtime in lead_lst:
            mon_arr, llst = lead_cnd(leadtime)
            for i, months in enumerate(llst):
                for space in space_lst:

                    if space == 'one':
                        one = loaded[:, :, mon_arr[i,0]:mon_arr[i,1], lat_lst[0]:lat_lst[1], lon_lst[0]:lon_lst[1]]
                        out = np.mean(one.reshape(model_num, year_num, leadtime*msk*msk), axis=2)

                    elif space == 'thailand':
                        thailand = loaded[:, :, mon_arr[i,0]:mon_arr[i,1], lat_lst[0]:lat_lst[1], lon_lst[0]:lon_lst[1]]
                        out = np.mean(thailand, axis=2)

                    elif space == 'monsoon':
                        monsoon = loaded[:, :, mon_arr[i,0]:mon_arr[i,1],  ma_lat[0]:ma_lat[1], ma_lon[0]:ma_lon[:]]
                        out = np.mean(monsoon, axis=2)

                    elif space == 'world':
                        world = loaded[:, :, mon_arr[i,0]:mon_arr[i,1], :, :]
                        out = np.mean(world, axis=2)

                    ofile = f"{workdir}/{variable}_{resolution}_{form}_{months}_{space}.npy"
                    np.save(ofile, out)
                    print(f"{ofile} is saved {out.shape}")

if __name__ == '__main__':
    main()

