# combinations:
# {'class': [5, 10, 30],
#  'type': [one, thailand, monsoon],
#  'month': [S, MJJASO],
#  'resolution': [1x1, 5x5],
#  'discrete': [EFD, EWD]} => 24*2
# try (5, one, MJJASO, 1x1, EFD) & (5, thailand, MJJASO, 5x5, EFD) => 2

import bisect
import numpy as np
from os.path import exists
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

def exec_resolution_discrete_class_space():
    class_list = [5, 10, 30]
    discrete_list = ['EFD']
    resolution_list = ['1x1', '5x5']
    space_list = ['monsoon']
    for s in space_list:
        for c in class_list:
            for d in discrete_list:
                for r in resolution_list:
                    main(class_num=c, discrete_mode=d, resolution=r, space=s)
    #plt.show()

def main(class_num=5, discrete_mode='EFD', resolution='1x1', space='monsoon'):
    # init
    ##########################################################################
    ##########################################################################
    ########################### most important switch ########################
    save_flag = True
    ##########################################################################
    ##########################################################################
    class_num = class_num
    discrete_mode = discrete_mode
    resolution = resolution
    if resolution == '1x1':
        key = '1x1'
        if space == 'thailand':
            lat_thailand, lon_thailand = 20, 20
        elif space == 'monsoon':
            lat_monsoon, lon_monsoon = 70, 90
    elif resolution == '5x5':
        key = '5x5_coarse'
        if space == 'thailand':
            lat_thailand, lon_thailand = 4, 4
        elif space == 'monsoon':
            lat_monsoon, lon_monsoon = 14, 18

    # path
    workdir = '/work/kajiyama/cnn/input/pr'

    if space == 'one':
        ### one <- thailand
        one_path = workdir + f"/continuous/one/{resolution}/pr_{key}_std_MJJASO_one.npy"
        one_spath = workdir + f"/class/one/{discrete_mode}" \
                    f"/pr_{key}_std_MJJASO_one_{discrete_mode}_{class_num}.npy"
        one = load(one_path) # one=(42, 165)

        if discrete_mode == 'EFD':
            # one_EFD
            one_class, one_bnd = one_EFD(one, class_num=class_num)
            print(f"one_bnd: {one_bnd}")
            save_npy(one_spath, one_class, save_flag=save_flag)
            draw_disc(one.reshape(42*165), one_bnd)

        elif discrete_mode == 'EWD':
            # one_EFD
            one_class, one_bnd = one_EWD(one, class_num=class_num)
            save_npy(one_spath, one_class, save_flag=save_flag)
            draw_disc(one.reshape(42*165), one_bnd)

    elif space == 'thailand':
        ### thailand
        fname = f"pr_{key}_std_MJJASO_thailand"
        thailand_path = workdir + f"/continuous/thailand/{resolution}/{fname}.npy"
        thailand_spath = workdir + f"/class/thailand/{discrete_mode}" \
                         f"/{fname}_{discrete_mode}_{class_num}.npy"
        thailand = load(thailand_path) # thailand=(42, 165, 4 , 4)

        if discrete_mode == 'EFD':
            # thailand_EFD
            thailand_class, thailand_bnd = thailand_EFD(thailand, class_num=class_num, lat_grid=lat_thailand, lon_grid=lon_thailand)
            print(f"thailand_bnd: {thailand_bnd}")
            save_npy(thailand_spath, thailand_class, save_flag=save_flag)
            draw_disc(thailand.reshape(42*165*lat_thailand*lon_thailand), thailand_bnd)
            show_thailand(thailand_class[0,0,:,:], class_num=class_num, lat_grid=lat_thailand, lon_grid=lon_thailand)

        elif discrete_mode == 'EWD':
            # one_EFD
            one_class, one_bnd = one_EWD(one, class_num=class_num)
            save_npy(one_spath, one_class, save_flag=save_flag)
            draw_disc(one.reshape(42*165), one_bnd)

            #thailand_EWD
            thailand_class, thailand_bnd = thailand_EWD(thailand, class_num=class_num, lat_grid=lat_thailand, lon_grid=lon_thailand)
            save_npy(thailand_spath, thailand_class, save_flag=save_flag)
            draw_disc(thailand.reshape(42*165*lat_thailand*lon_thailand), thailand_bnd)
            show_thailand(thailand_class[0,0,:,:], class_num=class_num, lat_grid=lat_thailand, lon_grid=lon_thailand)

    elif space == 'monsoon':
        ### monsoon
        fname_ma = f"pr_{key}_std_MJJASO_monsoon"
        monsoon_path = workdir + f"/continuous/monsoon/{resolution}/{fname_ma}.npy"
        monsoon_spath = workdir + f"/class/monsoon/{discrete_mode}" \
                         f"/{fname_ma}_{discrete_mode}_{class_num}.npy"
        monsoon = load(monsoon_path) # monsoon=(42, 165, 14 , 18)
        if discrete_mode == 'EFD':
            monsoon_class, monsoon_bnd = monsoon_EFD(monsoon, class_num=class_num, lat_grid=lat_monsoon, lon_grid=lon_monsoon)
            print(f"monsoon_bnd: {monsoon_bnd}")
            save_npy(monsoon_spath, monsoon_class, save_flag=save_flag)
            draw_disc(monsoon.reshape(42*165*lat_monsoon*lon_monsoon), monsoon_bnd)
            show_monsoon(monsoon_class[0,0,:,:], class_num=class_num, lat_grid=lat_monsoon, lon_grid=lon_monsoon)

def load(path):
    print(f"{path}: exist?=> {exists(path)}")
    npy = np.load(path)
    return npy

def save_npy(path, data, save_flag=False):
    if save_flag is True:
        np.save(path, data)
        print(f"class_output has been SAVED")
    else:
        print(f"class_output is ***NOT*** saved yet")

#############################################################################
###################### EFD conversion #######################################

def one_EFD(data, class_num=5):
    mjjaso_one = data.copy() # data=(42, 165)
    one_flat = mjjaso_one.reshape(42*165)
    flat_sorted = np.sort(one_flat)
    if len(flat_sorted)%class_num != 0:
        print('class_num is wrong')
    else:
        batch_sample = int(len(flat_sorted)/class_num)

    bnd = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    bnd.append(flat_sorted[-1] + 1e-10) # max boundary must be a bit higher than real max
    bnd[0] = bnd[0] - 1e-10 # min boundary must be a bit higher than real min
    bnd = np.array(bnd)
    one_class = np.empty(len(one_flat))
    for i, value in enumerate(one_flat):
        label = bisect.bisect(bnd, value)
        one_class[i] = int(label - 1)
    one_class.reshape(42, 165)
    return one_class, bnd

def thailand_EFD(data, class_num=5, lat_grid=4, lon_grid=4): # not-flattened input data required
    # EFD_bnd
    mjjaso_thailand = data.copy() # data=(42, 165, 4, 4)
    thailand_flat = mjjaso_thailand.reshape(42*165*lat_grid*lon_grid)
    flat_sorted = np.sort(thailand_flat)
    if len(flat_sorted)%class_num != 0:
        print('class_num is wrong')
    else:
        batch_sample = int(len(flat_sorted)/class_num)

    bnd = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    bnd.append(flat_sorted[-1] + 1e-10) # max boundary must be a bit higher than real max
    bnd[0] = bnd[0] -  1e-10 # min boundary must be a bit lower than real min
    bnd = np.array(bnd)

    # EFD_conversion
    thailand_class = np.empty(mjjaso_thailand.shape)
    for lat in  range(lat_grid):
        for lon in range(lon_grid):
            grid = mjjaso_thailand[:,:,lat,lon].reshape(42*165)
            grid_class = np.empty(len(grid))
            for i, value in enumerate(grid):
                label = bisect.bisect(bnd, value)
                grid_class[i] = int(label - 1)
            grid_class = grid_class.reshape(42, 165)
            thailand_class[:,:,lat,lon] = grid_class
    return thailand_class, bnd

def monsoon_EFD(data, class_num=5, lat_grid=14, lon_grid=18): # not-flattened input data required
    # EFD_bnd
    mjjaso_monsoon = data.copy() # data=(42, 165, 14, 18)
    monsoon_flat = mjjaso_monsoon.reshape(42*165*lat_grid*lon_grid)
    flat_sorted = np.sort(monsoon_flat)
    if len(flat_sorted)%class_num != 0:
        print('class_num is wrong')
    else:
        batch_sample = int(len(flat_sorted)/class_num)

    bnd = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    bnd.append(flat_sorted[-1] + 1e-10) # max boundary must be a bit higher than real max
    bnd[0] = bnd[0] -  1e-10 # min boundary must be a bit lower than real min
    bnd = np.array(bnd)

    # EFD_conversion
    monsoon_class = np.empty(mjjaso_monsoon.shape)
    for lat in  range(lat_grid):
        for lon in range(lon_grid):
            grid = mjjaso_monsoon[:,:,lat,lon].reshape(42*165)
            grid_class = np.empty(len(grid))
            for i, value in enumerate(grid):
                label = bisect.bisect(bnd, value)
                grid_class[i] = int(label - 1)
            grid_class = grid_class.reshape(42, 165)
            monsoon_class[:,:,lat,lon] = grid_class
    return monsoon_class, bnd

#############################################################################
###################### EWD conversion #######################################

def one_EWD(data, class_num=5):
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
    print(f"class_label: {u}")
    print(f"count: {counts}")
    print(f"bnd: {bnd}")
    print(f"max, min: {max(one_flat)}, {min(one_flat)}")
    return one_class, bnd # one_class=(6930), bnd=(class_num+1)

def thailand_EWD(data, class_num=5, lat_grid=4, lon_grid=4):
    mjjaso_thailand = data.copy() # data=(42, 165, 4, 4)
    thailand_flat = mjjaso_thailand.reshape(42*165*lat_grid*lon_grid)
    lim = max(abs(max(thailand_flat)), abs(min(thailand_flat)))
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
    thailand_class = np.empty(mjjaso_thailand.shape)
    for lat in  range(lat_grid):
        for lon in range(lon_grid):
            grid = mjjaso_thailand[:,:,lat,lon].reshape(42*165)
            grid_class = np.empty(len(grid))
            for i, value in enumerate(grid):
                label = bisect.bisect(bnd, value) # giving label
                grid_class[i] = int(label - 1)
            grid_class = grid_class.reshape(42, 165)
            thailand_class[:, :, lat, lon] = grid_class
    bnd = np.array(bnd)

    u, counts = np.unique(thailand_class, return_counts=True)
    print(f"class_label: {u}")
    print(f"count: {counts}")
    print(f"bnd: {bnd}")
    print(f"max, min: {max(thailand_flat)}, {min(thailand_flat)}")
    return thailand_class, bnd # thailand_class=(6930), bnd=(class_num+1)

#############################################################################
###################### draw tool ############################################

def draw_disc(data, bnd_list):
    """
    data shape must be one dimention,
    if one has (42, 65) shape, convert it to 42*165
    likewise, thailand has (42, 65, 4, 4), then convert it to 42*165*4*4
    """
    fig = plt.figure()
    ax = plt.subplot()
    ax.hist(data, bins=100, alpha=.5, color='darkcyan')
    for i in bnd_list:
        ax.axvline(i, ymin=0, ymax=len(data), alpha=.8, color='salmon')
    plt.show(block=False)

def show_thailand(image, class_num=5, lat_grid=4, lon_grid=4):
    cmap = plt.cm.get_cmap('BrBG', class_num)
    bounds = [i - 0.5 for i in range(class_num+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [i for i in range(class_num)]

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location=(N5-25, E90-110)
    txt_extent = (-88, -72, 7, 23) # location=(N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image, origin='upper', extent=img_extent, transform=projection, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mat, ax=ax, extend='both', ticks=ticks, spacing='proportional', orientation='vertical')

    if class_num == 5:
        cbar.ax.set_yticklabels(['low', 'mid-low', 'normal', 'mid-high', 'high'])
    elif class_num == 10:
        cbar.ax.set_yticklabels(['low', 'much-low', 'mid-low', 'little-low', 'normal',
                                 'normal', 'little-high', 'mid-high', 'much-high', 'high'])
    else:
        lat_lst = np.linspace(txt_extent[3], txt_extent[2], lat_grid)
        lon_lst = np.linspace(txt_extent[0], txt_extent[1], lon_grid)
        for i, lat in enumerate(lat_lst):
            for j, lon in enumerate(lon_lst):
                ax.text(lon, lat, image[i, j], 
                        ha="center", va="center", color='black', fontsize='15')
    plt.show(block=False)

def show_monsoon(image, class_num=5, lat_grid=14, lon_grid=18):
    cmap = plt.cm.get_cmap('BrBG', class_num)
    bounds = [i - 0.5 for i in range(class_num+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [i for i in range(class_num)]

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-120, -30, -15, 55) # location=(N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image, origin='upper', extent=img_extent, transform=projection, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mat, ax=ax, extend='both', ticks=ticks, spacing='proportional', orientation='vertical')

    if class_num == 5:
        cbar.ax.set_yticklabels(['low', 'mid-low', 'normal', 'mid-high', 'high'])
    plt.show(block=False)


if __name__ == '__main__':
    #main()
    exec_resolution_discrete_class_space()

