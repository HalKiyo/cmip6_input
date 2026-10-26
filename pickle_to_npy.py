import pickle
import sys

import numpy as np

WORKDIR = '/work/kajiyama/preprocessed/cmip6/'
SAVEDIR_FMT = '/work/kajiyama/cnn/input/{variable}/'

# 42 CMIP6 models, in the order of the model axis of every output array
MODELS = [
    'AWI-ESM-1-1-LR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CanESM5', 'CanESM5-CanOE',
    'CAS-ESM2-0', 'CESM2', 'CESM2-FV2', 'CESM2-WACCM', 'CMCC-CM2-HR4',
    'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1',
    'EC-Earth3', 'EC-Earth3-Veg-LR', 'FGOALS-f3-L', 'FGOALS-g3', 'GISS-E2-1-G',
    'GISS-E2-1-G-CC', 'GISS-E2-1-H', 'GISS-E2-2-G', 'GISS-E2-2-H', 'GISS-E3-G',
    'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'ICON-ESM-LR', 'IPSL-CM5A2-INCA', 'IPSL-CM6A-LR',
    'IPSL-CM6A-LR-INCA', 'MIROC6', 'MIROC-ES2H', 'MIROC-ES2L', 'MPI-ESM-1-2-HAM',
    'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1',
    'UKESM1-0-LL', 'UKESM1-1-LL',
]

# pickle key suffix and the per model shape of the stored array
FIELDS = [
    ('raw', (165, 12, 120, 360)),
    ('clim', (12, 120, 360)),
    ('variance', (12, 120, 360)),
    ('anom', (165, 12, 120, 360)),
    ('std', (165, 12, 120, 360)),
    ('coarse', (165, 12, 24, 72)),
    ('coarse_clim', (12, 24, 72)),
    ('coarse_variance', (12, 24, 72)),
    ('coarse_anom', (165, 12, 24, 72)),
    ('coarse_std', (165, 12, 24, 72)),
]

# accepted (val.shape[0], val.shape[1]) and the shape the value is reshaped to
# (None keeps it as is)
RESHAPE = {
    (1980, 120): (165, 12, 120, 360),
    (12, 120): None,
    (1980, 24): (165, 12, 24, 72),
    (12, 24): None,
}


def main():
    key_lst = ['mrso', 'pr', 'snc', 'tos', 'tsl'] # choose variable you want to store as much as you want as list object
    for key in key_lst:
        mk_newarr(key)


def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def mk_newarr(key):
    variable = key
    savedir = SAVEDIR_FMT.format(variable=variable)

    model_lst = [f"{WORKDIR}{m}/{variable}_{m}.pickle" for m in MODELS]
    variable_lst = [f"{variable}_{suffix}" for suffix, _ in FIELDS]
    arr_lst = [np.empty((len(model_lst),) + shape) for _, shape in FIELDS]
    sfile_lst = [f"{savedir}{valname}.npy" for valname in variable_lst]

    for i, modname in enumerate(model_lst):
        pkl = load(modname)
        print(f"{modname} is loaded")

        for j, valname in enumerate(variable_lst):
            val = pkl[valname]

            shape_key = (val.shape[0], val.shape[1])
            if shape_key not in RESHAPE:
                print(f"!!!ERROR!!! val.shape has wrong shape{val.shape}")
                sys.exit()
            new_shape = RESHAPE[shape_key]
            arr_lst[j][i] = val if new_shape is None else val.reshape(new_shape)
            print(f"{valname} is stored {val.shape}")

    for k, sfile in enumerate(sfile_lst):
        np.save(sfile, arr_lst[k])
        print(f"{sfile} is saved {arr_lst[k].shape}")


if __name__ == '__main__':
    main()
