import pickle
import numpy as np

def main():
    bnd_lst = ['pr'] # choose variable you want to store
    for bnd in bnd_lst:
        mk_newarr(bnd)

def load(file):
    with open (file, 'rb')  as f:
        data = pickle.load(f)
    return data


def mk_newarr(bnd):
    variable = bnd 
    workdir = '/work/kajiyama/preprocessed/cmip6/'
    savedir = f"/work/kajiyama/cnn/input/{variable}/"

    AWI = f"{workdir}AWI-ESM-1-1-LR/{variable}_AWI-ESM-1-1-LR.pickle"
    BMR = f"{workdir}BCC-CSM2-MR/{variable}_BCC-CSM2-MR.pickle"
    BM1 = f"{workdir}BCC-ESM1/{variable}_BCC-ESM1.pickle"
    Ca5 = f"{workdir}CanESM5/{variable}_CanESM5.pickle"
    CaE = f"{workdir}CanESM5-CanOE/{variable}_CanESM5-CanOE.pickle"
    CAS = f"{workdir}CAS-ESM2-0/{variable}_CAS-ESM2-0.pickle"
    CE2 = f"{workdir}CESM2/{variable}_CESM2.pickle"
    CV2 = f"{workdir}CESM2-FV2/{variable}_CESM2-FV2.pickle"
    CCM = f"{workdir}CESM2-WACCM/{variable}_CESM2-WACCM.pickle"
    CR4 = f"{workdir}CMCC-CM2-HR4/{variable}_CMCC-CM2-HR4.pickle"
    CR5 = f"{workdir}CMCC-CM2-SR5/{variable}_CMCC-CM2-SR5.pickle"
    CM2 = f"{workdir}CMCC-ESM2/{variable}_CMCC-ESM2.pickle"
    CN1 = f"{workdir}CNRM-CM6-1/{variable}_CNRM-CM6-1.pickle"
    CNR = f"{workdir}CNRM-CM6-1-HR/{variable}_CNRM-CM6-1-HR.pickle"
    CN2 = f"{workdir}CNRM-ESM2-1/{variable}_CNRM-ESM2-1.pickle"
    EC3 = f"{workdir}EC-Earth3/{variable}_EC-Earth3.pickle"
    ECR = f"{workdir}EC-Earth3-Veg-LR/{variable}_EC-Earth3-Veg-LR.pickle"
    FGL = f"{workdir}FGOALS-f3-L/{variable}_FGOALS-f3-L.pickle"
    FG3 = f"{workdir}FGOALS-g3/{variable}_FGOALS-g3.pickle"
    GIG = f"{workdir}GISS-E2-1-G/{variable}_GISS-E2-1-G.pickle"
    GIC = f"{workdir}GISS-E2-1-G-CC/{variable}_GISS-E2-1-G-CC.pickle"
    GIH = f"{workdir}GISS-E2-1-H/{variable}_GISS-E2-1-H.pickle"
    G2G = f"{workdir}GISS-E2-2-G/{variable}_GISS-E2-2-G.pickle"
    G2H = f"{workdir}GISS-E2-2-H/{variable}_GISS-E2-2-H.pickle"
    G3G = f"{workdir}GISS-E3-G/{variable}_GISS-E3-G.pickle"
    HLL = f"{workdir}HadGEM3-GC31-LL/{variable}_HadGEM3-GC31-LL.pickle"
    HMM = f"{workdir}HadGEM3-GC31-MM/{variable}_HadGEM3-GC31-MM.pickle"
    ICR = f"{workdir}ICON-ESM-LR/{variable}_ICON-ESM-LR.pickle"
    I5A = f"{workdir}IPSL-CM5A2-INCA/{variable}_IPSL-CM5A2-INCA.pickle"
    I6R = f"{workdir}IPSL-CM6A-LR/{variable}_IPSL-CM6A-LR.pickle"
    I6A = f"{workdir}IPSL-CM6A-LR-INCA/{variable}_IPSL-CM6A-LR-INCA.pickle"
    MC6 = f"{workdir}MIROC6/{variable}_MIROC6.pickle"
    M2H = f"{workdir}MIROC-ES2H/{variable}_MIROC-ES2H.pickle"
    M2L = f"{workdir}MIROC-ES2L/{variable}_MIROC-ES2L.pickle"
    MPM = f"{workdir}MPI-ESM-1-2-HAM/{variable}_MPI-ESM-1-2-HAM.pickle"
    MPH = f"{workdir}MPI-ESM1-2-HR/{variable}_MPI-ESM1-2-HR.pickle"
    MPL = f"{workdir}MPI-ESM1-2-LR/{variable}_MPI-ESM1-2-LR.pickle"
    MRI = f"{workdir}MRI-ESM2-0/{variable}_MRI-ESM2-0.pickle"
    NMM = f"{workdir}NorESM2-MM/{variable}_NorESM2-MM.pickle"
    Tai = f"{workdir}TaiESM1/{variable}_TaiESM1.pickle"
    U0L = f"{workdir}UKESM1-0-LL/{variable}_UKESM1-0-LL.pickle"
    U1L = f"{workdir}UKESM1-1-LL/{variable}_UKESM1-1-LL.pickle"

    model_lst = [AWI, BMR, BM1, Ca5, CaE, CAS, CE2, CV2, CCM, CR4, CR5, CM2, CN1, CNR, 
                 CN2, EC3, ECR, FGL, FG3, GIG, GIC, GIH, G2G, G2H, G3G, HLL, HMM, ICR, 
                 I5A, I6R, I6A, MC6, M2H, M2L, MPM, MPH, MPL, MRI, NMM, Tai, U0L, U1L]

    variable_lst = [f"{variable}_raw", f"{variable}_clim", f"{variable}_variance",
                    f"{variable}_anom", f"{variable}_std",
                    f"{variable}_coarse", f"{variable}_coarse_clim", f"{variable}_coarse_variance",
                    f"{variable}_coarse_anom", f"{variable}_coarse_std"]

    intgrtd_raw =              np.empty((len(model_lst), 165, 12, 120, 360))
    intgrtd_clim =             np.empty((len(model_lst), 12, 120, 360))
    intgrtd_variance =         np.empty((len(model_lst), 12, 120, 360))
    intgrtd_anom =             np.empty((len(model_lst), 165, 12, 120, 360))
    intgrtd_std =              np.empty((len(model_lst), 165, 12, 120, 360))
    intgrtd_coarse =           np.empty((len(model_lst), 165, 12, 24, 72))
    intgrtd_coarse_clim =      np.empty((len(model_lst), 12, 24, 72))
    intgrtd_coarse_variance =  np.empty((len(model_lst), 12, 24, 72))
    intgrtd_coarse_anom =      np.empty((len(model_lst), 165, 12, 24, 72))
    intgrtd_coarse_std =       np.empty((len(model_lst), 165, 12, 24, 72))

    arr_lst = [intgrtd_raw, intgrtd_clim, intgrtd_variance, 
               intgrtd_anom, intgrtd_std, 
               intgrtd_coarse, intgrtd_coarse_clim, intgrtd_coarse_variance,
               intgrtd_coarse_anom, intgrtd_coarse_std]

    raw_path =             f"{savedir}{variable}_raw.npy"
    clim_path =            f"{savedir}{variable}_clim.npy"
    variance_path =        f"{savedir}{variable}_variance.npy"
    anom_path =            f"{savedir}{variable}_anom.npy"
    std_path =             f"{savedir}{variable}_std.npy"
    coarse_path =          f"{savedir}{variable}_coarse.npy"
    coarse_clim_path =     f"{savedir}{variable}_coarse_clim.npy"
    coarse_variance_path = f"{savedir}{variable}_coarse_variance.npy"
    coarse_anom_path =     f"{savedir}{variable}_coarse_anom.npy"
    coarse_std_path =      f"{savedir}{variable}_coarse_std.npy"

    sfile_lst = [raw_path, clim_path, variance_path,
                 anom_path, std_path,
                 coarse_path, coarse_clim_path, coarse_variance_path,
                 coarse_anom_path, coarse_std_path]

    for i, modname in enumerate(model_lst):
        pkl = load(modname)
        print(f"{modname} is loaded")

        for j, valname in enumerate(variable_lst):
            val = pkl[valname]

            if val.shape[0] == 1980 and val.shape[1] == 120:
                intgrtd = val.reshape(165, 12, 120, 360)
                arr_lst[j][i,:,:,:,:] = intgrtd
                print(f"{valname} is stored {val.shape}")
            elif val.shape[0] == 12 and val.shape[1] == 120:
                intgrtd = val
                arr_lst[j][i,:,:,:] = intgrtd
                print(f"{valname} is stored {val.shape}")
            elif val.shape[0] == 1980 and val.shape[1] == 24:
                intgrtd = val.reshape(165, 12, 24, 72)
                arr_lst[j][i,:,:,:] = intgrtd
                print(f"{valname} is stored {val.shape}")
            elif val.shape[0] == 12 and val.shape[1] == 24:
                intgrtd = val
                arr_lst[j][i,:,:,:] = intgrtd
                print(f"{valname} is stored {val.shape}")
            else:
                print(f"!!!ERROR!!! val.shape has wrong shape{val.shape}")
                exit()

    for k, sfile in enumerate(sfile_lst):
        np.save(sfile, arr_lst[k])
        print(f"{sfile} is saved {arr_lst[k].shape}")


if __name__ == '__main__':
    main()

