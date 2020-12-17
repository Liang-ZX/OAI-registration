from math import sqrt
import pandas as pd
import numpy as np
from MyProcessDicom import *
import SimpleITK as sitk


def get_max_coordinate(image):
    array = sitk.GetArrayFromImage(image)
    return np.max(np.nonzero(np.argmax(np.argmax(array, axis = 1), axis=1)))


def computeQualityMeasures(lP,lT):
    quality=dict()
#     labelPred=sitk.GetImageFromArray(lP, isVector=False)
#     labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    labelPred = lP
    labelTrue = lT
    space = labelPred.GetSpacing()[2]
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue,labelPred)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance() * space
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance() * space
 
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue,labelPred)
    quality["dice"]=dicecomputer.GetDiceCoefficient()
    
    quality["max_y_coordinate"]= (get_max_coordinate(labelPred) - get_max_coordinate(labelTrue)) * space
 
    return quality


def compute_quality_metrics(all_image_data):
    last_name = '0000000L'
    avgHausdorff, Hausdorff, dice, max_y_coordinate, count = {}, {}, {}, {}, {}
    # for each image
    for i in range(0, len(all_image_data)):
        image_data                    = all_image_data[i]
        tpid = image_data['moving_name'][9:12]
        if(image_data['moving_name'][12]=="_"):
#             print(image_data['moving_name'][:12])
            pass
        else:
            tpid += image_data['moving_name'][12]
#             print(image_data['moving_name'][:13])
   
        reference_file_name = "C:\\Zhixuan\\OAI-registration\\pykneer-yg\\reference\\longitudinal\\" + image_data['fmask'][:-5] + 't.mha'
        tibia_file_name = image_data['segmented_folder'] + image_data['fmask'][:-5] + 't_rigid.mha'
#         moving_file_name   = image_data['moving_folder']    + image_data['moving_name']
        ref = sitk.ReadImage(reference_file_name)
        tibia = sitk.ReadImage(tibia_file_name)
        ref = resample_bwimage(ref)
        tibia = resample_bwimage(tibia)
        quality = computeQualityMeasures(tibia,ref)
        avgHausdorff[tpid] = avgHausdorff.get(tpid, 0) + quality['avgHausdorff']
        Hausdorff[tpid] = Hausdorff.get(tpid, 0) + quality['Hausdorff']
        dice[tpid] = dice.get(tpid, 0) + quality['dice']
        max_y_coordinate[tpid] = max_y_coordinate.get(tpid, 0) + abs(quality['max_y_coordinate'])
        count[tpid] = count.get(tpid, 0) + 1
    ret =  {'avgHausdorff': avgHausdorff, 'Hausdorff': Hausdorff, 'dice': dice, 'max_y_coordinate': max_y_coordinate, 'count': count}
#     print(ret)
    for key in avgHausdorff:
        ret['avgHausdorff'][key] = avgHausdorff[key] / count[key]
        ret['Hausdorff'][key] = Hausdorff[key] / count[key]
        ret['dice'][key] = dice[key] / count[key]
        ret['max_y_coordinate'][key] = max_y_coordinate[key] / count[key]
    print(ret)
    return ret


def computeCenterLineMeasures(ref_path, pred_path):
    df = pd.read_csv(pred_path, header=None, sep=' ', names=['x','y','z'])
    line = np.array([df['x'], df['y'], df['z']]).T
    df2 = pd.read_csv(ref_path, header=1, sep=' ', names=['x','y','z'])
    line2 = np.array([df2['x'], df2['y'], df2['z']]).T

    j = 0
    dist, cnt = 0, 0
    for i in range(line.shape[0]):
        tmp_z = line[i][2]
        if tmp_z < line2[j][2] and int(tmp_z) != line2[j][2]:
            continue
        while tmp_z >= line2[j][2] and j < line2.shape[0]-1:
            j = j + 1
        if j >= line2.shape[0]:
            break
        if (line2[j][2] - tmp_z) < (tmp_z - line2[j-1][2]):
            kk = j
        else:
            kk = j - 1
        dist += sqrt((line[i][0] - line2[kk][0])**2 + (line[i][1] - line2[kk][1])**2)
        cnt += 1
    print(dist/cnt, cnt)
    return dist/cnt


def compute_centerline_metrics(all_image_data):
    dis_dict,count = {}, {}
    for i in range(0, len(all_image_data)):
        image_data                    = all_image_data[i]
        tpid = image_data['moving_name'][9:12]
        if(image_data['moving_name'][12]=="_"):
            print(image_data['moving_name'][:12], end = ':\t')
        else:
            tpid += image_data['moving_name'][12]
            print(image_data['moving_name'][:13], end = ':\t')
   
        reference_line_name = "C:\\Zhixuan\\OAI-registration\\pykneer-yg\\reference\\centerline\\" + image_data['fmask'][:-10] + 'line.txt'
        predict_line_name = 'C:\\Zhixuan\\OAI-registration\\pykneer-yg\\centerline\\' + image_data['fmask'][:-10] + 'line.txt'
#         moving_file_name   = image_data['moving_folder']    + image_data['moving_name']
        
        dist = computeCenterLineMeasures(reference_line_name, predict_line_name)
        dis_dict[tpid] = dis_dict.get(tpid, 0) + dist
        count[tpid] = count.get(tpid, 0) + 1
    for key in dis_dict:
        dis_dict[key] /= count[key]
    print(dis_dict)
    return
