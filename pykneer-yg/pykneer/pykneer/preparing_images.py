import pydicom
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import importlib
import sys
import cv2
import SimpleITK as sitk

sys.path.append(r'C:\Zhixuan\FRAPPE')
import DB
importlib.reload(DB)
from DB import DB
import CASCADE
importlib.reload(CASCADE)
from CASCADE import CASCADE
import glob

TPS = ['0','12','18','24','30','36','48','60','72','84','96']
sys.path.append(r'C:\Zhixuan\OAI-registration')
from ServerDicomIO import getdcmpath
from icafeAPI import generate_centerline


def drawcont(cas, shape, fi):
    vesmask = np.zeros(shape,dtype=np.uint8)
                        
    cont = cas.getcontour('OuterWall',fi+1)
    if cont is None or np.size(cont[0])<2:
        print('No lumen contour for frappe dcm slice',fi)
        return vesmask      
    
    intcont = []
    for conti in range(len(cont)):
        intcont.append([int(round(cont[conti][0])),int(round(cont[conti][1]))])
    intcont = np.array(intcont)	
    cv2.fillPoly(vesmask, pts=[intcont], color=(1,1,1))
    return vesmask   


def flip_rl(img):

    # flip the image
    flip_direction = [True, False, False]
    img = sitk.Flip(img, flip_direction)

    # put direction back to identity (direction(0,0) becomes -1 after flip)
    direction = []
    direction.append(1.0)
    direction.append(0.0)
    direction.append(0.0)
    direction.append(0.0)
    direction.append(1.0)
    direction.append(0.0)
    direction.append(0.0)
    direction.append(0.0)
    direction.append(1.0)
    img.SetDirection(direction)
    
    img.SetOrigin((0,0,0))

    return img


def prepare_image_and_list(caselist):

    split_seq_path = r'..\..\original'
    vesmaskpath = r'..\..\vesmask'

    if not os.path.exists(split_seq_path):
        os.mkdir(split_seq_path)

    if not os.path.exists(vesmaskpath):
        os.mkdir(vesmaskpath)

   #split seqs for a case

    # create .\image_list_preprocessing.txt
    file_path = "image_list_preprocessing.txt"
    list_file = open(file_path, 'w+')
    newline = '..\..\original\\' + '\n'
    list_file.write(newline)

    # create .\image_list_newsubject.txt
    file_path2 = "image_list_newsubject.txt"
    subject_file = open(file_path2, 'w+')
    newline = 'C:\Zhixuan\OAI-registration\pyKNEEr-yg\\reference\\newsubject' + '\n'
    subject_file.write(newline)
    newline = 'C:\Zhixuan\OAI-registration\pyKNEEr-yg\preprocessed' + '\n'
    subject_file.write(newline)
    newline = 'r reference.mha\n'
    subject_file.write(newline)

    # create .\image_list_longitudinal.txt
    file_path3 = "image_list_longitudinal.txt"
    long_file = open(file_path3, 'w+')
    newline = 'C:\Zhixuan\OAI-registration\pyKNEEr-yg\\reference\longitudinal' + '\n'
    long_file.write(newline)
    newline = 'C:\Zhixuan\OAI-registration\pyKNEEr-yg\preprocessed' + '\n'
    long_file.write(newline)
    
    # create .\image_list_reference.txt
    file_path4 = "image_list_reference.txt"
    ref_file = open(file_path4, 'w+')
    newline = 'C:\Zhixuan\OAI-registration\pyKNEEr-yg\\reference\\newsubject' + '\n'
    ref_file.write(newline)
    newline = 'C:\Zhixuan\OAI-registration\pyKNEEr-yg\preprocessed' + '\n'
    ref_file.write(newline)
    newline = 'r reference.mha\n'
    ref_file.write(newline)

    for casei in caselist:
        pi = casei['pid']
        side = casei['side']

        precasepath = split_seq_path + '/' + pi + side + '/'
        if not os.path.exists(precasepath):
            os.mkdir(precasepath)

        regtp = casei['TP']
        regtp2 = casei['TP2']

        # modify .\image_list_newsubject.txt
        newline = 'm '+pi+side+'_TP0_prep.mha\n'
        subject_file.write(newline)

        for tpi in range(len(regtp)):
            casepath = getdcmpath(pi, regtp[tpi], side)

            if casepath is None:
                print('cannot find dcm path for TPid',regtp[tpi])
                continue
            else:
                print('dcm path for TPid', regtp[tpi], casepath)

                tp_precasepath = precasepath + 'TP' + str(regtp[tpi]) + '/'
                if not os.path.exists(tp_precasepath):
                    os.mkdir(tp_precasepath)

                # modify .\image_list_preprocessing.txt
                newline = pi + side + '\TP' + str(regtp[tpi]) + '\n'
                list_file.write(newline)
                if side == 'L':
                    tmpline = 'left\n'
                else:
                    tmpline = 'right\n'
                list_file.write(tmpline)

                # modify .\image_list_longitudinal.txt
                if regtp[tpi] == 0:
                    newline = 'r ' + pi + side + '_TP' + str(regtp[tpi]) + '_prep.mha' + '\n'
                else:
                    newline = 'm ' + pi + side + '_TP' + str(regtp[tpi]) + '_prep.mha' + '\n'
                    ref_file.write(newline)
                long_file.write(newline)

                filelist = os.listdir(casepath)
                for f in filelist:
                    if ('.dcm') not in f:
                        del filelist[filelist.index(f)]

                eidir = r'Y:/'+str(regtp2[tpi])+'tp29\cascade/'+str(regtp2[tpi])+'/'+pi+'/*'+side # In \\128.208.221.24\OAI-Derived-Data-and-Results
                glb = glob.glob(eidir)
                if len(glb)==0:
                    print('No ei')
                    continue

                ei = os.path.basename(glb[0])
                casname = 'E'+ei+'_'+side
                cascadepath = r'Y:/'+str(regtp2[tpi])+'tp29\cascade/'+str(regtp2[tpi])+'/'+pi+'/'+ei
                cas = CASCADE(casname,cascadepath,cascadepath,0)

                tmp_path = vesmaskpath + '/'+pi+side+'_TP'+str(regtp[tpi])+'_prep_fv.mha'  # modify
                if os.path.exists(tmp_path):
                    continue
                
                #very ugly here
                vesmaskstack = []
                for fi in range(len(filelist)):
                    src = casepath + '/' + filelist[fi]

                    srcdcm = pydicom.dcmread(src)
                    if abs(float(srcdcm.ImageOrientationPatient[0]))<abs(float(srcdcm.ImageOrientationPatient[1])):
                        #coronal slice showing scan areas
                        print('coronal',src)
                        continue

                    for attempt in range(20):
                        try:
                            dst = tp_precasepath + filelist[fi]
                            if not os.path.exists(dst):
                                shutil.copyfile(src, dst)
                                print('\rcopying', src, end='')
                        except:
                            print('@@@@Connection loss. Retry!@@@@')
                            continue
                        break
                    else:
                        print('###Tried 20 times, still failed')

                    vesmask = drawcont(cas, srcdcm.pixel_array.shape, fi)
                    vesmaskstack.append(vesmask)
                vesmaskstack = np.array(vesmaskstack,dtype=np.uint8)

                vesmasksitk = sitk.GetImageFromArray(vesmaskstack)
                tmpsrcdcm = pydicom.dcmread(casepath + '/' + filelist[len(filelist)//2])
                vesmasksitk.SetSpacing((tmpsrcdcm[0x0028,0x0030].value[0],tmpsrcdcm[0x0028,0x0030].value[1],tmpsrcdcm[0x0018,0x0050].value))

                if side == 'R':
                    vesmasksitk = flip_rl(vesmasksitk)
                
                sitk.WriteImage(vesmasksitk, tmp_path)
                if regtp[tpi] == 0:
                    shutil.copy(tmp_path, "C:\\Zhixuan\\OAI-registration\\pykneer-yg\\reference\\longitudinal")

    list_file.close()
    subject_file.close()
    long_file.close()
    ref_file.close()
    
    
def prepare_centerline(caselist):
    ref_dir = r"C:\\Zhixuan\\OAI-registration\\pykneer-yg/reference/centerline/"
    for case in caselist:
        pid = case['pid']
        side = case['side']
        for j in range(len(case['TP'])):
            if (generate_centerline(pid, case['TP'][j], side)):
                file_path = r'C:\\Zhixuan\\centerline/P'+pid+side+'/tracing_raw_ves_TH_'+str(case['TP'][j])+'_P'+pid+side+'_U.swc'
                ref_path = ref_dir + pid + side + "_TP" + str(case['TP'][j]) + "_line.txt"
                df = pd.read_csv(file_path, header=None, sep=' ', names=['index', 'vessel_id','x','y','z','undefined','last_id'])
                line = np.array([df['x'], df['y'], df['z']]).T
#                 df['x'] = 384 - df['x']
#                 df['y'] = 384 - df['y']
#                 df['z'] = 4.11 * 75 - df['z']
                with open(ref_path,"w+") as f:
                    f.write("point\n")
                    f.write(str(line.shape[0])+"\n")
                df.to_csv(ref_path,sep=" ",columns=['x','y','z'], mode='a', header=None, index=False)

                
def modify_output_line(tmp_path, write_path):
    wfile = open(write_path,"w+")
    for line in open(tmp_path,"r"): 
        index = line.find('OutputPoint')
        index1 = line.find('[', index, len(line))
        index2 = line.find(']', index, len(line))
        num_str = line[index1+1:index2]
        wordlist=num_str.split()
        for i, a in enumerate(wordlist):
            wfile.write(a)
            if i == 2:
                wfile.write('\n')
            else:
                wfile.write(' ')
    wfile.close()               

    