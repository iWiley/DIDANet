import os
import multiprocessing
import warnings
import logging
import SimpleITK as sitk
import tempfile
import pandas as pd
import datetime
import pickle
import shutil
from tqdm import tqdm
from radiomics import featureextractor 

warnings.filterwarnings("ignore")
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
DIR_VOL_IN = '/DIR_VOL_IN'
DIR_MASK_IN = '/DIR_MASK_IN'
DIR_OUT = '/DIR_OUT'
DIR_OUT_ROOT = '/DIR_OUT_ROOT'
MAX_THREADS = None
ID_LABEL_BACKGROUND = 1
ID_LABEL_TUMOR = 2
ID_LABEL_MARGIN = 3
MASK_MARCH = True

def catch_features(extractor, imagePath, maskPath, case, p, num, total):
    if MASK_MARCH:
        img = sitk.ReadImage(imagePath)
        mask = sitk.ReadImage(maskPath)
        mask = sitk.Resample(mask,
                    referenceImage=img,
                    transform =sitk.Transform(),
                    interpolator=sitk.sitkNearestNeighbor,
                    defaultPixelValue = 0,
                    outputPixelType = mask.GetPixelID())
        size = mask.GetSize()
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2] - 4, size[2]):
                    mask.SetPixel(x,y,z,0)
        f = tempfile.NamedTemporaryFile()
        fn = f'{f.name}.nii.gz'
        f.close()
        wriiter = sitk.ImageFileWriter()
        wriiter.SetFileName(fn)
        wriiter.Execute(mask)
        maskPath = fn
    r_b = extractor.execute(imagePath, maskPath, ID_LABEL_BACKGROUND)
    print(f'\033[47;30m Note \033[0m {num}/{total} {case} {p} ID_LABEL_BACKGROUND Extraction success')
    r_t = extractor.execute(imagePath, maskPath, ID_LABEL_TUMOR)
    print(f'\033[47;30m Note \033[0m {num}/{total} {case} {p} ID_LABEL_TUMOR Extraction success')
    r_m = extractor.execute(imagePath, maskPath, ID_LABEL_MARGIN)
    print(f'\033[47;30m Note \033[0m {num}/{total} {case} {p} ID_LABEL_MARGIN Extraction success')
    if MASK_MARCH:
        os.remove(fn)
    return r_b, r_t, r_m

def checkPath(case, num, total):
    volumes = ['AP', 'VP', 'DP']
    i = 0
    paths = []
    for vol in volumes:
        i += 1
        pVol = f'{DIR_VOL_IN}/{case}/{i}.{vol}.nii.gz'
        if not os.path.exists(pVol):
            print(f'\033[43;30m Warning \033[0m {num}/{total} The processing of {case} has been skipped because the sequence file [{i}. {vol}] does not exist, processing of {case} has been skipped.')
            return False, paths
        pMask = f'{DIR_MASK_IN}/{case}/{i}.{vol}.Segmentation-label-segmentation-label.nrrd'
        if not os.path.exists(pMask):
            pMask = [f for f in os.listdir(f'{DIR_MASK_IN}/{case}') if f.startswith(f'{i}.{vol}.Segmentation') and f.endswith('segmentation-label.nrrd')]
            if not len(pMask) == 0:
                pMask =f'{DIR_MASK_IN}/{case}/{pMask[0]}'
            else:
                pMask = f'{DIR_MASK_IN}/{case}/{i}.{vol}.Segmentation-label-segmentation-label.nii.gz'
                if not os.path.exists(pMask):
                    pMask = [f for f in os.listdir(f'{DIR_MASK_IN}/{case}') if f.startswith(f'{i}.{vol}.Segmentation') and f.endswith('segmentation-label.nii.gz')]
                    if not len(pMask) == 0:
                        pMask = f'{DIR_MASK_IN}/{case}/{pMask[0]}'
                    else:
                        print(f'\033[43;30m Warning \033[0m {num}/{total} The processing of {case} has been skipped because the mask file [{i}. {vol}] does not exist, processing of {case} has been skipped.')
                        return False, paths
        paths.append((pVol, pMask, vol))
    return True, paths

def invoke(case, ns, num, total):
    exists, paths = checkPath(case, num, total)
    if not exists:
        return
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    print(f'\033[47;30m Note \033[0m {num}/{total} {case} Begin extraction.')
    cols = []
    values = []
    i = 0
    for p, p_m, vol in paths:
        i += 1
        try:
            b, t, m = catch_features(extractor, p, p_m, case, vol, num, total)
        except (Exception, BaseException) as e:
            b, t, m = None, None, None
            print(f'\033[41m Error \033[0m {num}/{total} Error on {p} of {case}, {e}')
            print(f'\033[43;30m Warning \033[0m {num}/{total} Processing of the remaining sequence of {case} has been skipped due to an error in the previous sequence.')
            return
        print(f'\033[46m Success \033[0m {num}/{total} {case} Extract {p} Success')
        for tissue, result in [('B', b), ('T', t), ('M', m)]:
            x = 0
            for k, v in result.items():
                if x >= 22:
                    cols.append(f'{vol}_{tissue}_{k}')
                    values.append(v)
                x += 1
    
    with open(f'{ns.tmp}/{case}.col', 'wb') as f:
        pickle.dump(cols, f)
    with open(f'{ns.tmp}/{case}.val', 'wb') as f:
        pickle.dump(values, f)
    print(f'\033[42m Success \033[0m {num}/{total} {case} Extraction is complete and the number of cases has been successfully extracted:{len(os.listdir(ns.tmp)) / 2:.0f}。')

if __name__ == '__main__':
    if os.path.exists(f'{DIR_MASK_IN}/settings'):
        print(f'\033[47;30m Success \033[0m The settings file is detected and the default configuration will be overwritten!')
        with open(f'{DIR_MASK_IN}/settings','r+') as set:
            settings = eval(set.read()) 
    else:
        settings = {}
        settings['binWidth'] = 25 
        settings['sigma'] = [3, 5]
        settings['voxelArrayShift'] = 1000 
        settings['normalize'] = True
        settings['normalizeScale'] = 100
        settings['geometryTolerance'] = 1e-05 
    total = [f.name for f in os.scandir(DIR_MASK_IN) if f.is_dir()]
    if len(total) == 0:
        total = [""]
    mgr = multiprocessing.Manager()
    ns = mgr.Namespace()
    tmp = f'{DIR_OUT}/tmp'
    if os.path.exists(tmp) == False:
        os.mkdir(tmp)
    ns.tmp = tmp
    file = f'{DIR_OUT}/Radiomics-Features-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}.csv'
    err_call_back = lambda e: print(f'\033[41m 错误 \033[0m {e}')
    pool = multiprocessing.Pool(MAX_THREADS)
    num = 0
    for case in total:
        num += 1
        if os.path.exists(f'{tmp}/{case}.col') == False or os.path.exists(f'{tmp}/{case}.val') == False:
            pool.apply_async(invoke, (case, ns, num, len(total),), error_callback=err_call_back)
        else:
            print(f'\033[47;30m Note \033[0m {case} has been skipped')
    pool.close()
    pool.join()
    print('\033[47;30m Note \033[0m Please wait...')
    error = []
    df = pd.DataFrame()
    bar = tqdm(total=len(total), desc="Writing", unit='case')
    for case in total:
        if os.path.exists(f'{tmp}/{case}.col') == False or os.path.exists(f'{tmp}/{case}.val') == False:
            error.append(case)
            bar.update()
            continue
        with open(f'{tmp}/{case}.col', 'rb') as f:
            col = pickle.load(f)
        with open(f'{tmp}/{case}.val', 'rb') as f:
            val = pickle.load(f)
        if len(df.index) == 0:
            df = pd.DataFrame(columns=col)
        df.loc[case] = val
        bar.update()
    bar.close()
    df.to_csv(file)
    shutil.copy(file, f'{DIR_OUT_ROOT}/Radiomics-Features.csv')
    if len(error) == 0:
        print(f'\033[42m Success \033[0m The task was successfully completed and the document has been saved to {file}。')
    else:
        print(f'\033[42m Success \033[0m The task was successfully completed and the document has been saved to {file}. The following sequence extraction failed:')
        print(*error, sep=',')