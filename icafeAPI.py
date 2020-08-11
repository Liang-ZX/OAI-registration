import os
import importlib
import sys
sys.path.append(r'../FRAPPE')
import DB
importlib.reload(DB)
from DB import DB

import Func
importlib.reload(Func)
from Func import icafeswc


def geteid(pid,TPid,side='L'):
    #following TPS
    TPS = ['0','12','18','24','30','36','48','60','72','84','96']
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
    eis = db.geteidbyside((TPS[TPid],pid,side))
    if not len(eis):
        return None
    else:
        ei = eis[0][0]
        print('TP',TPid,'found ei',ei)
        return ei

def generate_centerline(pid, TPid, side='L'):
    TPS = ['0','12','18','24','30','36','48','60','72','84','96']
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

    eis = db.geteidbyside((TPS[TPid],pid,side))
    if not len(eis):
        return
    eid = eis[0][0]
    spacingbetweenslices = 1.5
    pixelspacing = 0.36458
    bbswclist = db.getSWCresult((TPS[TPid],pid,eid))
    dir_path = r'../result/OAIMTP/P'+pid+side
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    swcname = dir_path+'/tracing_raw_ves_TH_'+str(TPid)+'_P'+pid+side+'_U.swc'
    swdname = dir_path+'/tracing_raw_ves_TH_'+str(TPid)+'_P'+pid+side+'_U.swd'
    icafeswc(bbswclist,swcname,swdname,spacingbetweenslices/pixelspacing)
    return