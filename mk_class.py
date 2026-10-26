# combinations:
# {'class': [5, 10, 30],
#  'type': [one, thailand, monsoon],
#  'month': [S, MJJASO],
#  'resolution': [1x1, 5x5],
#  'discrete': [EFD, EWD]} => 24*2
# try (5, one, MJJASO, 1x1, EFD) & (5, thailand, MJJASO, 5x5, EFD) => 2

from os.path import exists

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from discrete import classify, efd_bnd, ewd_bnd, ewd_report

##########################################################################
########################### most important switch ########################
SAVE_FLAG = True
##########################################################################

WORKDIR = '/work/kajiyama/cnn/input/pr'

# file name key for each resolution
RES_KEY = {'1x1': '1x1', '5x5': '5x5_coarse'}

# (lat_grid, lon_grid) for each space and resolution
GRID = {
    'thailand': {'1x1': (20, 20), '5x5': (4, 4)},
    'monsoon': {'1x1': (70, 90), '5x5': (14, 18)},
}

TICKLABELS_5 = ['low', 'mid-low', 'normal', 'mid-high', 'high']
TICKLABELS_10 = ['low', 'much-low', 'mid-low', 'little-low', 'normal',
                 'normal', 'little-high', 'mid-high', 'much-high', 'high']


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
    key = RES_KEY[resolution]
    fname = f"pr_{key}_std_MJJASO_{space}"
    path = WORKDIR + f"/continuous/{space}/{resolution}/{fname}.npy"
    spath = WORKDIR + f"/class/{space}/{discrete_mode}" \
                      f"/{fname}_{discrete_mode}_{class_num}.npy"

    if space == 'one':
        ### one <- thailand
        one = load(path)  # one=(42, 165)

        if discrete_mode == 'EFD':
            one_class, one_bnd = one_EFD(one, class_num=class_num)
            print(f"one_bnd: {one_bnd}")
            save_npy(spath, one_class, save_flag=SAVE_FLAG)
            draw_disc(one.reshape(42*165), one_bnd)

        elif discrete_mode == 'EWD':
            one_class, one_bnd = one_EWD(one, class_num=class_num)
            save_npy(spath, one_class, save_flag=SAVE_FLAG)
            draw_disc(one.reshape(42*165), one_bnd)

    elif space == 'thailand':
        lat_grid, lon_grid = GRID[space][resolution]
        thailand = load(path)  # thailand=(42, 165, lat_grid, lon_grid)

        if discrete_mode == 'EFD':
            thailand_class, thailand_bnd = thailand_EFD(thailand, class_num=class_num, lat_grid=lat_grid, lon_grid=lon_grid)
            print(f"thailand_bnd: {thailand_bnd}")
        elif discrete_mode == 'EWD':
            thailand_class, thailand_bnd = thailand_EWD(thailand, class_num=class_num, lat_grid=lat_grid, lon_grid=lon_grid)
        else:
            return
        save_npy(spath, thailand_class, save_flag=SAVE_FLAG)
        draw_disc(thailand.reshape(42*165*lat_grid*lon_grid), thailand_bnd)
        show_thailand(thailand_class[0,0,:,:], class_num=class_num, lat_grid=lat_grid, lon_grid=lon_grid)

    elif space == 'monsoon':
        lat_grid, lon_grid = GRID[space][resolution]
        monsoon = load(path)  # monsoon=(42, 165, lat_grid, lon_grid)

        if discrete_mode == 'EFD':
            monsoon_class, monsoon_bnd = monsoon_EFD(monsoon, class_num=class_num, lat_grid=lat_grid, lon_grid=lon_grid)
            print(f"monsoon_bnd: {monsoon_bnd}")
            save_npy(spath, monsoon_class, save_flag=SAVE_FLAG)
            draw_disc(monsoon.reshape(42*165*lat_grid*lon_grid), monsoon_bnd)
            show_monsoon(monsoon_class[0,0,:,:], class_num=class_num, lat_grid=lat_grid, lon_grid=lon_grid)


def load(path):
    print(f"{path}: exist?=> {exists(path)}")
    npy = np.load(path)
    return npy


def save_npy(path, data, save_flag=False):
    if save_flag is True:
        np.save(path, data)
        print("class_output has been SAVED")
    else:
        print("class_output is ***NOT*** saved yet")

#############################################################################
###################### EFD conversion #######################################


def one_EFD(data, class_num=5):
    one_flat = data.reshape(42*165)  # data=(42, 165)
    bnd = efd_bnd(one_flat, class_num=class_num)
    one_class = classify(one_flat, bnd)
    return one_class, bnd  # one_class=(6930), bnd=(class_num+1)


def grid_EFD(data, class_num=5, lat_grid=4, lon_grid=4):  # not flattened input data required
    flat = data.reshape(42*165*lat_grid*lon_grid)  # data=(42, 165, lat_grid, lon_grid)
    bnd = efd_bnd(flat, class_num=class_num)
    return classify(data, bnd), bnd


def thailand_EFD(data, class_num=5, lat_grid=4, lon_grid=4):
    return grid_EFD(data, class_num=class_num, lat_grid=lat_grid, lon_grid=lon_grid)


def monsoon_EFD(data, class_num=5, lat_grid=14, lon_grid=18):
    return grid_EFD(data, class_num=class_num, lat_grid=lat_grid, lon_grid=lon_grid)

#############################################################################
###################### EWD conversion #######################################


def one_EWD(data, class_num=5):
    one_flat = data.reshape(42*165)  # data=(42, 165)
    bnd = ewd_bnd(one_flat, class_num=class_num)
    one_class = classify(one_flat, bnd)
    ewd_report(one_class, bnd, one_flat)
    return one_class, bnd  # one_class=(6930), bnd=(class_num+1)


def thailand_EWD(data, class_num=5, lat_grid=4, lon_grid=4):
    thailand_flat = data.reshape(42*165*lat_grid*lon_grid)  # data=(42, 165, lat_grid, lon_grid)
    bnd = ewd_bnd(thailand_flat, class_num=class_num)
    thailand_class = classify(data, bnd)
    ewd_report(thailand_class, bnd, thailand_flat)
    return thailand_class, bnd  # thailand_class=data.shape, bnd=(class_num+1)

#############################################################################
###################### draw tool ############################################


def draw_disc(data, bnd_list):
    """
    data shape must be one dimention,
    if one has (42, 65) shape, convert it to 42*165
    likewise, thailand has (42, 65, 4, 4), then convert it to 42*165*4*4
    """
    plt.figure()
    ax = plt.subplot()
    ax.hist(data, bins=100, alpha=.5, color='darkcyan')
    for i in bnd_list:
        ax.axvline(i, ymin=0, ymax=len(data), alpha=.8, color='salmon')
    plt.show(block=False)


def _show_map(image, class_num, img_extent):
    """Draw a class map on PlateCarree(central_longitude=180) and return (ax, cbar)."""
    cmap = plt.get_cmap('BrBG', class_num)
    bounds = [i - 0.5 for i in range(class_num+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [i for i in range(class_num)]

    projection = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image, origin='upper', extent=img_extent, transform=projection, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mat, ax=ax, extend='both', ticks=ticks, spacing='proportional', orientation='vertical')
    return ax, cbar


def show_thailand(image, class_num=5, lat_grid=4, lon_grid=4):
    img_extent = (-90, -70, 5, 25)  # location=(N5-25, E90-110)
    txt_extent = (-88, -72, 7, 23)  # location=(N5-25, E90-110)
    ax, cbar = _show_map(image, class_num, img_extent)

    if class_num == 5:
        cbar.ax.set_yticklabels(TICKLABELS_5)
    elif class_num == 10:
        cbar.ax.set_yticklabels(TICKLABELS_10)
    else:
        lat_lst = np.linspace(txt_extent[3], txt_extent[2], lat_grid)
        lon_lst = np.linspace(txt_extent[0], txt_extent[1], lon_grid)
        for i, lat in enumerate(lat_lst):
            for j, lon in enumerate(lon_lst):
                ax.text(lon, lat, image[i, j],
                        ha="center", va="center", color='black', fontsize='15')
    plt.show(block=False)


def show_monsoon(image, class_num=5, lat_grid=14, lon_grid=18):
    img_extent = (-120, -30, -15, 55)  # location=(S15-N55, E60-150)
    ax, cbar = _show_map(image, class_num, img_extent)

    if class_num == 5:
        cbar.ax.set_yticklabels(TICKLABELS_5)
    plt.show(block=False)


if __name__ == '__main__':
    #main()
    exec_resolution_discrete_class_space()
