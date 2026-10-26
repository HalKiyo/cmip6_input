# input variance = [mrso, snc, tos, tsl]; 4C1 + 4C2 + 4C3 + 4C4 = 4+6+4+1=15
# predictors representations: m:mrso, s:snc, o:tos, t:tsl
# variable_key_month_predictors
# example = 'predictors_coarse_std_Aug_ ***|*** msot.npy'

from itertools import combinations

import numpy as np

WORKDIR = '/work/kajiyama/cnn/input'
KEYS = ['coarse_std', 'coarse_anom', 'coarse', 'std', 'anom', 'raw']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
VARIABLES = ['mrso', 'snc', 'tos', 'tsl']
LETTERS = {'mrso': 'm', 'snc': 's', 'tos': 'o', 'tsl': 't'}

# all 15 combinations in the order m, s, o, t, ms, mo, mt, so, st, ot, mso, mst, mot, sot, msot
SELECTION = [list(c) for r in range(1, len(VARIABLES) + 1) for c in combinations(VARIABLES, r)]
CMBS = [''.join(LETTERS[v] for v in slct) for slct in SELECTION]


def main():
    for key in KEYS:
        data = {val: np.load(f"{WORKDIR}/{val}/{val}_{key}.npy") for val in VARIABLES}
        for i, mon in enumerate(MONTHS):
            for c, slct in zip(CMBS, SELECTION):
                predictors = np.stack([data[v][:,:,i,:,:] for v in slct])
                ofile = f"{WORKDIR}/predictors/predictors_{key}_{mon}_{c}.npy"
                np.save(ofile, predictors)
                print(f"{ofile} is saved {predictors.shape}; month_i:{i}")


if __name__ == '__main__':
    main()
