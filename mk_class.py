# combinations:
# {'class': [5, 10, 100],
#  'type': [one, thailand],
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

def main():
    workdir = '/work/kajiyama/cnn/input/pr'
    one_path = workdir + '/one/1x1/pr_1x1_std_MJJASO_one.npy'
    thailand_path = workdir + '/thailand/5x5/pr_5x5_coarse_std_MJJASO_thailand.npy'

    one = load(one_path)
    one_flat = one.reshape(42*165)
    thailand = load(thailand_path)

    thailand_class, thailand_bnd = thailand_EFD(thailand, class_num=5)
    print(f"thailand_bnd: {thailand_bnd}")
    show_class(thailand_class[0,0,:,:], class_num=5)

def load(path):
    print(f"path existance: {exists(path)}")
    npy = np.load(path)
    return npy

def thailand_EFD(data, class_num=5): # not-flattened input data required
    # EFD_bnd
    mjjaso_thailand = data.copy() # data=(42, 165, 4, 4)
    thailand_flat = mjjaso_thailand.reshape(42*165*4*4)
    flat_sorted = np.sort(thailand_flat)
    if len(flat_sorted)%class_num != 0:
        print('class_num is wrong')
    else:
        batch_sample = int(len(flat_sorted)/class_num)

    bnd = [flat_sorted[i] for i in range(0, len(flat_sorted), batch_sample)]
    bnd.append(flat_sorted[-1])
    bnd = np.array(bnd)

    # EFD_trans
    thailand_class = np.empty(mjjaso_thailand.shape)
    for lat in  range(4):
        for lon in range(4):
            grid = mjjaso_thailand[:,:,lat,lon].reshape(42*165)
            grid_class = np.empty(len(grid))
            for i, value in enumerate(grid):
                label = bisect.bisect(bnd, value)
                grid_class[i] = int(label - 1)
            grid_class = grid_class.reshape(42, 165)
            thailand_class[:,:,lat,lon] = grid_class
    return thailand_class, bnd

def EFD(data, class_num=5):
    out = data.copy() # data=(6930)
    out_sorted = np.sort(out)
    if len(data)%class_num != 0:
        print('class-num is wrong')
    else:
        batch_sample = int(len(data)/class_num)

    out_bnd = [out_sorted[i] for i in range(0, len(out_sorted), batch_sample)]
    out_class = np.empty(len(out_sorted))
    for i, value in enumerate(out):
        label = bisect.bisect(out_bnd, value)
        out_class[i] = int(label-1)

    out_bnd.append(out_sorted[-1])
    out_bnd = np.array(out_bnd)
    u, counts = np.unique(out_class, return_counts=True)
    return out_class, out_bnd # out_class=(6930), out_bnd=(class_num+1)

def EWD(data, class_num=5):
    out = data.copy() # data=(6930)
    lim = max(abs(max(data)), abs(min(data)))
    dx = 2*lim/class_num

    out_bnd = []
    out_bnd.append(-lim)
    out_bnd.append(lim)
    if class_num%2 == 0:
        origin = 0
        out_bnd.append(origin)
    else:
        origin = dx/2
        out_bnd.append(origin)
        out_bnd.append(-origin)

    loop_num = int(class_num/2)
    for i in range(loop_num):
        out_bnd.append(origin+dx*(i+1))
        out_bnd.append(-origin-dx*(i+1))
    out_bnd = np.sort(out_bnd)

    out_class = np.empty(len(out))
    for i, value in enumerate(out):
        label = bisect.bisect(out_bnd, value) # giving label
        out_class[i] = int(label - 1)
    out_bnd = np.array(out_bnd)

    u, counts = np.unique(out_class, return_counts=True)
    return out_class, out_bnd # out_class=(6930), out_bnd=(class_num+1)

def draw_disc(data, bnd_list):
    fig = plt.figure()
    ax = plt.subplot()
    ax.hist(data, bins=1000, alpha=.5, color='darkcyan')
    for i in bnd_list:
        ax.axvline(i, ymin=0, ymax=len(data), alpha=.8, color='salmon')
    plt.show()

def show_class(image, class_num=5):
    cmap = plt.cm.get_cmap('BrBG', class_num)
    bounds = [i - 0.5 for i in range(class_num+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [i for i in range(class_num)]

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location=(N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image, origin='upper', extent=img_extent, transform=projection, norm=norm, cmap=cmap)
    cbar = fig.colorbar(mat, ax=ax, extend='both', ticks=ticks, spacing='proportional', orientation='vertical')
    cbar.ax.set_yticklabels(['low', 'mid-low', 'normal', 'mid-high', 'high'])
    plt.show()


if __name__ == '__main__':
    main()

