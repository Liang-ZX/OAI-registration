import pydicom
import os,re
import numpy as np
import shutil
import cv2
from pydicom.dataset import Dataset, FileDataset
import tempfile
import SimpleITK as sitk
import math
from skimage.transform import resize
from scipy.ndimage import zoom
import scipy.misc
import importlib
import sys
sys.path.append(r'..\FRAPPE')
import DB
importlib.reload(DB)
from DB import DB

icafepath = r'Y:/iCafe/result/OAIMTP/'
transform_path = r'../transform/'
TPS = ['0','12','18','24','30','36','48','60','72','84','96']
split_seq_path = 'split_seq_OAI/'
if not os.path.exists(split_seq_path):
    os.mkdir(split_seq_path)

def getdcmpath(pid,TPid,side='L'):
    #following TPs
    VFVersion = '29'
    paths = []
    if TPid not in [0,1,2,3,4,5,6,8,10]:
        print('not valid TP')
        return 
        
    dbconfig = {}
    dbconfig['dbname'] = 'ahaknee'+TPS[TPid]+'tp'+VFVersion
    dbconfig['host']="128.208.221.46"#Server #4
    dbconfig['user']="root"
    dbconfig['passwd']="123456"
    db = DB(config=dbconfig)

    db.getcursor()
    sql = '''\
    SELECT
    pid,eid,dicompath
    FROM
    stat
    where pid = %s
    '''
    db.mycursor.execute(sql,[pid])
    dbr = db.mycursor.fetchall() 
    for di in dbr:
        if di[1][-1]==side:
            tmp = di[2]
            return 'Z' + tmp[1:]
    return


#split seqs for a case
def getFileFromDB(caselist):    
    for casei in caselist:
        pi = casei['pid']
        side = casei['side']

        precasepath = split_seq_path + pi + side + '/'
        if not os.path.exists(precasepath):
            os.mkdir(precasepath)

        regtp = casei['TP']

        for tpi in regtp:
            print('********  tp %d ************' % tpi)
            casepath = getdcmpath(pi, tpi, side)
            
            if casepath is None:
                print('cannot find dcm path for TPid',tpi)
                continue
            else:
                print('dcm path for TPid', tpi, casepath)

                tp_precasepath = precasepath + 'TP' + str(tpi) + '/'
                if not os.path.exists(tp_precasepath):
                    os.mkdir(tp_precasepath)

                filelist = os.listdir(casepath)
                for f in filelist:
                    if ('.dcm') not in f:
                        del filelist[filelist.index(f)]

                for f in filelist:
                    src = casepath + '/' + f
                    print('\rcopying', src, end=' ')
                    for attempt in range(20):
                        try:
                            dst = tp_precasepath + f
                            if not os.path.exists(dst):
                                shutil.copyfile(src, dst)

                        except:
                            print('@@@@Connection loss. Retry!@@@@')
                            continue
                        break
                    else:
                        print('###Tried 20 times, still failed')
                print()
    return


def get_dicom_series(tp1_casepath):
    TP1path = []
    flist = os.listdir(tp1_casepath)
    for f in flist:
        dicomfilename = tp1_casepath + f
        dcmimgcr = pydicom.dcmread(dicomfilename)
        if abs(float(dcmimgcr.ImageOrientationPatient[0]))<abs(float(dcmimgcr.ImageOrientationPatient[1])):
            #coronal slice showing scan areas
            print('coronal',dicomfilename)
            os.remove(dicomfilename)
        else:
            TP1path.append(dicomfilename)
    
    fixed_image = sitk.ReadImage(TP1path)
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    fixed_image = resacleFilter.Execute(fixed_image)
    return fixed_image