# input variance = [mrso, snc, tos, tsl]; 4C1 + 4C2 + 4C3 + 4C4 = 4+6+4+1=15
# predictors representations: m:mrso, s:snc, o:tos, t:tsl
# variable_key_month_predictors
# example = 'predictors_coarse_std_Aug_ ***|*** msot.npy' 

import numpy as np

def main():
    workdir = '/work/kajiyama/cnn/input'
    keys = ['coarse_std', 'coarse_anom', 'coarse', 'std', 'anom', 'raw']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cmbs = ['m', 's', 'o', 't',
            'ms', 'mo', 'mt', 'so', 'st', 'ot',
            'mso', 'mst', 'mot', 'sot',
            'msot']
    variables = ['mrso', 'snc', 'tos', 'tsl']
    val_dict = {'mrso': 0, 'snc': 1, 'tos': 2, 'tsl': 3}
    selection = [
                 ['mrso'], ['snc'], ['tos'], ['tsl'],
                 ['mrso', 'snc'], ['mrso', 'tos'], ['mrso', 'tsl'], ['snc', 'tos'], ['snc', 'tsl'], ['tos', 'tsl'],
                 ['mrso', 'snc', 'tos'], ['mrso', 'snc', 'tsl'], ['mrso', 'tos', 'tsl'], ['snc', 'tos', 'tsl'],
                 ['mrso', 'snc', 'tos', 'tsl']
                ]

    for key in keys:
        data = [ np.load( f"{workdir}/{val}/{val}_{key}.npy" ) for val in variables ]
        for i, mon in enumerate(months):
            for c, slct in zip(cmbs, selection):
                index = [ val_dict[s] for s in slct ]
                value = [data[ind] for ind in index]
                predictors = np.stack( [v[:,:,i,:,:] for v in value ])
                ofile = f"{workdir}/predictors/predictors_{key}_{mon}_{c}.npy"
                np.save(ofile, predictors)
                print(f"{ofile} is saved {predictors.shape}; month_i:{i}")

if __name__ == '__main__':
    main()

