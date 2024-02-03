import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

DIR_IN = "/DIR_IN"
DIR_IN_VOL = "/DIR_IN_VOL"
DIR_OUT = "/DIR_OUT"
DEBUG = False

def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], interpolator = sitk.sitkBSpline):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(itk_image)
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.BSplineTransform(3))
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(interpolator)
    return resample.Execute(itk_image)
def Process(file:str, save: str, interpolator):
    if file == None:
        return
    if os.path.exists(save):
        tqdm.write(f'File already exists: {file}, skip...')
        return
    Original_img = sitk.ReadImage(file)
    if DEBUG:
        tqdm.write(f'Original image parameters: [Spacing] {Original_img.GetSpacing()}, [Size] {Original_img.GetSize()}')
    Resample_img = resample_image(Original_img, interpolator = interpolator)
    if DEBUG:
        tqdm.write(f'Parameters after resampling: [Spacing] {Resample_img.GetSpacing()}, [Size] {Resample_img.GetSize()}')
    wriiter = sitk.ImageFileWriter()
    wriiter.SetFileName(save)
    wriiter.Execute(Resample_img)
def checkPath(case, v):
    pVol = f'{DIR_IN}/{case}/{v}.nii.gz'
    if os.path.exists(pVol) == False:
        if os.path.exists(f'{DIR_IN_VOL}/{case}/{v}'):
            pVol = f'{DIR_IN_VOL}/{case}/{v}'
    pMask = f'{DIR_IN}/{case}/{v}.Segmentation-label.nrrd'
    if not os.path.exists(pMask):
        pMask = [f for f in os.listdir(f'{DIR_IN}/{case}') if f.startswith(f'{v}.Segmentation') and f.endswith('.nrrd')]
        if not len(pMask) == 0:
            pMask =f'{DIR_IN}/{case}/{pMask[0]}'
        else:
            pMask = f'{DIR_IN}/{case}/{v}.Segmentation-label.nii.gz'
            if not os.path.exists(pMask):
                pMask = [f for f in os.listdir(f'{DIR_IN}/{case}') if f.startswith(f'{v}.Segmentation') and f.endswith('.nii.gz')]
                if not len(pMask) == 0:
                    pMask = f'{DIR_IN}/{case}/{pMask[0]}'
                else:
                    pMask = None
    if not os.path.exists(pVol):
        pVol = None
        tqdm.write(f'\033[41mFile does not exist\033 The sequence file {v} for {case} does not exist and has been skipped.')
    if pMask == None:
        tqdm.write(f'\033[41mFile does not exist\033 The mask file {v} for {case} does not exist, skipped.')
    return pVol, pMask

if __name__ == '__main__':
    total = [f.name for f in os.scandir(DIR_IN) if f.is_dir()]
    if len(total) == 0:
        total = [""]
    out_spacing = [1.0, 1.0, 1.0]
    i = 0
    with tqdm(total=len(total) * 3, unit = 'case') as pbar:
        for case in total:
            i = i + 1
            AP_Vol, AP_Mask = checkPath(case, '1.AP')
            VP_Vol, VP_Mask = checkPath(case, '2.VP')
            DP_Vol, DP_Mask = checkPath(case, '3.DP')
            O_AP = f'{DIR_OUT}/{case}/1.AP.nii.gz'
            O_AP_Mask = f'{DIR_OUT}/{case}/1.AP.Segmentation-label.nii.gz'
            O_VP = f'{DIR_OUT}/{case}/2.VP.nii.gz'
            O_VP_Mask = f'{DIR_OUT}/{case}/2.VP.Segmentation-label.nii.gz'
            O_DP = f'{DIR_OUT}/{case}/3.DP.nii.gz'
            O_DP_Mask = f'{DIR_OUT}/{case}/3.DP.Segmentation-label.nii.gz'
            if os.path.exists(f'{DIR_OUT}/{case}') == False:
                os.mkdir(f'{DIR_OUT}/{case}')
            if os.path.exists(O_AP):
                tqdm.write(f"{O_AP}Already exists, skip")
            else:
                Process(AP_Vol, O_AP, sitk.sitkBSpline1)

            if os.path.exists(O_AP_Mask):
                tqdm.write(f"{O_AP_Mask}Already exists, skip")
            else:
                Process(AP_Mask, O_AP_Mask, sitk.sitkLabelGaussian)
            pbar.update(1)

            if os.path.exists(O_VP):
                tqdm.write(f"{O_VP}Already exists, skip")
            else:
                Process(VP_Vol, O_VP, sitk.sitkBSpline1)
                
            if os.path.exists(O_VP_Mask):
                tqdm.write(f"{O_VP_Mask}Already exists, skip")
            else:
                Process(VP_Mask, O_VP_Mask, sitk.sitkLabelGaussian)
            pbar.update(1)

            if os.path.exists(O_DP):
                tqdm.write(f"{O_DP}Already exists, skip")
            else:
                Process(DP_Vol, O_DP, sitk.sitkBSpline1)
            
            if os.path.exists(O_DP_Mask):
                tqdm.write(f"{O_DP_Mask}Already exists, skip")
            else:
                Process(DP_Mask, O_DP_Mask, sitk.sitkLabelGaussian)
            pbar.update(1)