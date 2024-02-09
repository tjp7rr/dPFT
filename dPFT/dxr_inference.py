# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 07:52:09 2022

@author: tpisano



#can set anaconda ipython console path to C:\ProgramData\Anaconda3\envs\sleap\python.exe

#conda create -y -n sleap -c sleap -c sleap/label/dev -c nvidia -c conda-forge sleap=1.2.0a6
conda create -y -n sleap -c sleap -c nvidia -c conda-forge sleap=1.2.1
conda activate sleap

#seems to be issues with pydicom installations esp when using pip (might need to conda install pip)
conda install -c conda-forge pydicom
pip install imageio-ffmpeg


#set spyder ipython console to initialize under enviroment: tools>preferences>python interpreter
conda install spyder-kernels=2.1
C:\ProgramData\Anaconda3\envs\sleap\python.exe

###
###alternatives###
https://github.com/murthylab/sleap/releases
https://github.com/murthylab/sleap/issues/614
#conda create -y -n sleap_v1.1.5 -c sleap sleap=1.1.5
#conda create -y -n sleap_v1.1.4 -c sleap sleap=1.1.4


#Make separate file, add vidoes of interest, add labeling suggestions (like 275/video), run inference, clear all labeling suggestions, then add labels of interest, merge into other file

"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
tpisano

sleap.ai

source activate sleap
sleap-label

#tracking
sleap-track

#NEED TO GET FRAME RATE OF VIDEOS*****

ffmpeg -y -i "video.avi" -c:v libx264 -crf 25 -preset superfast -pix_fmt yuv420p "video.sf.mp4"

"""
import matplotlib
import os, pandas as pd, numpy as np, shutil, scipy, pywt, cv2, datetime, subprocess as sp, skimage
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
import imageio, h5py
import warnings, gc
import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from scipy.interpolate import Rbf
#from skimage.external import tifffile

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
pd.pandas.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Keep; these are definitions used by functions. Order is IMPORTANT*****
llung = ['L Lat D','L9', 'L8', 'L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'LC CW', 'L Med CW', 'LU Med CW', 'L Med D', 'L RU D', 'L Cen D', 'L LU D']
rlung = ['R Lat D','R9', 'R8', 'R7', 'R6', 'R5', 'R4', 'R3', 'R2', 'RC CW', 'RU Med CW', 'R Med CW', 'R Med D', 'R LU D', 'R Cen D', 'R RU D'] #this is slightly dif order than L
heart = ['S AK', 'LU AK', 'L AK', 'LL AK', 'AK PT', 'PT', 'PT AtAp', 'AtAp', 'AtAp LV', 'LV LU', 'Lat LV', 'LL LV', 'RL RV', 'Lat RV', 'RU RV', 'RV AsAo', 'AsAo', 'RU AsAo', 'RU AArch']
shoulders = ['L Coracoid','L Dist Clav','L Mid Clav','L Prox Clav','L SternClav', 'R SternClav', 'R Prox Clav', 'R Mid Clav', 'R Dist Clav', 'R Coracoid']

llung_lheart = ['L Lat D','L9', 'L8', 'L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'LC CW', 'L Med CW', 'LU Med CW', #lateral lung before diaphragm
                      'S AK', 'LU AK', 'L AK', 'LL AK', 'AK PT', 'PT', 'PT AtAp', 'AtAp', 'AtAp LV', 'LV LU', 'Lat LV', 'LL LV',#heart border
                      #'L Cen D', 'L LU D'] #if you only want the part of L lung diaphragm...,
                        'L Med D', 'L RU D', 'L Cen D', 'L LU D'] #if you want medial part of diphragm then use this on this line:

rlung_rheart = ['R Lat D','R9', 'R8', 'R7', 'R6', 'R5', 'R4', 'R3', 'R2', 'RC CW', 'RU Med CW', 'R Med CW', #lateral lung before diaphragm
                      'RU AsAo', 'AsAo', 'RV AsAo', 'RU RV', 'Lat RV', 'RL RV',#heart border
                      'R Med D', 'R LU D', 'R Cen D', 'R RU D'] #remainer of lung diaphragm
total_lung = ['L Lat D','L9', 'L8', 'L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'LC CW', 'L Med CW', 'LU Med CW',
              'R Med CW','RU Med CW','RC CW','R2','R3','R4','R5','R6','R7','R8','R9','R Lat D','R RU D','R Cen D','R LU D','R Med D',
              'L Med D', 'L RU D', 'L Cen D', 'L LU D']
rdiaphragm = ['R Lat D', 'R RU D', 'R Cen D', 'R LU D', 'R Med D'] #lateral to medial
ldiaphragm = ['L Lat D', 'L LU D', 'L Cen D', 'L RU D', 'L Med D'] #lateral to medial


def makedir(src):
    if not os.path.exists(src): os.mkdir(src)
    return
def removedir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    return

def remove_empty_folders(src):
    folders = list(os.walk(src))[1:]

    for folder in folders:
        if not folder[2]:
            try:
                os.rmdir(folder[0])
            except:pass
    return

def warn_fxn():
    warnings.warn("deprecated", DeprecationWarning)
    return

def dicom2arr(path, voi_lut=True, fix_monochrome=True, subtract_value=0.0):
    dicom = pydicom.read_file(path)
    data = apply_voi_lut(dicom.pixel_array, dicom) if (voi_lut) else dicom.pixel_array
    if (fix_monochrome and (dicom.PhotometricInterpretation == "MONOCHROME1")):
        data = np.amax(data) - data
    data = data.astype(np.float)
    data -= np.min(data)
    data /= np.max(data)
    data = np.maximum(data-subtract_value,0) #TP
    data = (data * 255).astype(np.uint8)

    return data

def point_distance(xy0, xy1):
    return np.sqrt( (xy0[0] - xy1[0])**2 + (xy0[1] - xy1[1])**2 )


def shortest_distance(p1, p2, p3):
    ''' Function to find perpindicular distance distance between line p0-p1 and point p2'''
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

def get_line(x1, y1, x2, y2):
    '''
    from https://stackoverflow.com/questions/25837544/get-all-points-of-a-straight-line-in-python
    '''
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

def give_points_on_line(x1,y1,x2,y2,val=0, no_of_pnts=4):
    '''
    val is just a value that it taks on

    '''
    pnts = [(float(x1),float(y1), float(val))] #return the original points
    for ii in range(no_of_pnts):
        pnts.append((
            float(min(x1,x2)+int((ii+1)*(abs(x1 - x2)/(no_of_pnts+1)))),
                     float(min(y1,y2) + int((ii+1)*(abs(y1 - y2)/(no_of_pnts+1)))),
                     float(val)))
    pnts.append((float(x2),float(y2),float(val)))
    return pnts

def give_points_on_line_wrapper(x0,y0,x1,y1, mean_change, no_of_pnts=1):
    data=[]
    for i in range(x0.shape[0]):
        data.append([give_points_on_line(x0[i], y0[i], x1[i], y1[i], no_of_pnts=no_of_pnts, val=mean_change[i])])
    d = np.asarray([zz for xx in data for yy in xx for zz in yy])
    return np.asarray(d)

def norm_arr(data, subtract_value=0.0):

    data = data.astype(np.float)
    data -= np.min(data)
    data /= np.max(data)
    data = np.maximum(data-subtract_value,0) #TP
    data = (data * 255).astype(np.uint8)

    return data

def interpolator_2d(x,y,z,xmin=0,xmax=1072,ymin=0,ymax=1072, inter_type='IDW'):
    '''
    inter_type = 'IDW', 'RBF', 'linRBF'
    modified from
    https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python

    '''
    x = np.squeeze(x);y = np.squeeze(y);z = np.squeeze(z);
    nx, ny = xmax, ymax
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    # Calculate IDW
    if inter_type=='IDW':
        grid1 = simple_idw(x,y,z,xi,yi, power=5)
        grid1 = grid1.reshape((ny, nx))
        return grid1

    elif inter_type=='RBF':
        # Calculate scipy's RBF
        grid2 = scipy_idw(x,y,z,xi,yi)
        grid2 = grid2.reshape((ny, nx))
        return grid2

    elif inter_type=='linRBF':
        #this often breaks
        grid3 = linear_rbf(x,y,z,xi,yi)
        #print (grid3.shape)
        grid3 = grid3.reshape((ny, nx))
        return grid3

def simple_idw(x, y, z, xi, yi, power=1):
    """ from https://gist.github.com/Majramos/5e8985adc467b80cccb0cc22d140634e
    Simple inverse distance weighted (IDW) interpolation
    Weights are proportional to the inverse of the distance, so as the distance
    increases, the weights decrease rapidly.
    The rate at which the weights decrease is dependent on the value of power.
    As power increases, the weights for distant points decrease rapidly.
    """
    dist = distance_matrix(x,y, xi,yi)
    # In IDW, weights are 1 / distance
    weights = 1.0/(dist+1e-12)**power
    # Make weights sum to one
    weights /= weights.sum(axis=0)
    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)

def linear_rbf(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)
    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)
    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)
    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    return zi
def scipy_idw(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)
def distance_matrix(x0, y0, x1, y1):
    """ Make a distance matrix between pairwise observations.
    Note: from <http://stackoverflow.com/questions/1871536>
    """
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    # calculate hypotenuse
    return np.hypot(d0, d1)
def nninterpolator(xss,yss,vss,xmin=0,xmax=1075,ymin=0,ymax=1075, inter_type='IDW'):
    ''' this works similiarly to IDW above, but much slower
    '''
    from scipy.interpolate import NearestNDInterpolator
    points = np.stack([yss,xss],axis=1)
    myInterpolator = NearestNDInterpolator(points, vss)

    data = np.zeros((ymax,xmax))
    for i in range(ymax):
        for j in range(xmax):
            data[i,j]=myInterpolator(i,j)

def listdirfull(src, keyword=False):
    if not keyword: fls = [(src + '/' + xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx]
    if keyword: fls = [(src + '/' + xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx and keyword in xx]
    fls.sort()
    return fls

def listall(fld, keyword=False):
    '''function to recursively list all files within a folder and subfolders
    Inputs:
    -----------------------
    fld = folder to search
    '''
    fls = []
    for (root, dir, files) in os.walk(fld):
         for f in files:
             path = root + '/' + f
             if os.path.exists(path):
                 fls.append(path)

    if keyword: fls = [xx for xx in fls if keyword in xx]

    #windows hack
    fls = [xx.replace('\\','/') for xx in fls]

    return fls

def flipLR(mp4name):
    vid = imageio.get_reader(mp4name)
    arr = np.asarray([i[1] for i in enumerate(vid)])
    arr = np.flip(arr, 2) ##flip and save it
    imageio.mimwrite(mp4name, arr, fps = vid.get_meta_data()['fps'], **{'macro_block_size':16})
    return


def convert_to_mp4(fl):
    import subprocess as sp
    nfl = fl+'.mp4' if fl[-4:]!='.mp4' else fl
    sp.check_output('ffmpeg -y -i "{}" -c:v libx264 -crf 25 -preset superfast -pix_fmt yuv420p "{}"'.format(fl, nfl), shell=True, stderr=sp.STDOUT, stdin=sp.DEVNULL)
    return

def read_mp4_to_arr(src):
    
    reader = imageio.get_reader(src)
    fps = reader.get_meta_data()['fps']
    arr = []
    for im in reader:
        arr.append(im)
        
    return np.asarray(arr)

def SLEAP_inference(model, datapth, dst, tracker=False, batchsize=8, enviroment_name = 'sleap', gpu=True, givecall=False):
    #sleap-track --model E:/dynamic_xray/sleap/models/210412_142752.single_instance.n=88 -o E:/dynamic_xray/sleap_output/No0005-Original_PA_Deep.avi.predictions.slp --verbosity rich --batch_size 1 --tracking.tracker flow E:/dynamic_xray/data01/Data/No0005/No0005-Original_PA_Deep.avi
    outfl = dst + '/' + os.path.basename(datapth) + ".predictions.slp"

    if type(model)==list:

        if tracker: call = ['sleap-track', '--model', model[0], '--model', model[1], '-o', outfl,
             "--verbosity", "rich", '--batch_size', str(batchsize), '--tracking.tracker', 'flow', datapth]

        #removing flow tracker
        if not tracker: call = ['sleap-track','--model', model[0], '--model', model[1], '-o', outfl,
                 "--verbosity", "rich", '--batch_size', str(batchsize), datapth]

    else:

        if tracker: call = ['sleap-track', '--model', model, '-o', outfl,
                 "--verbosity", "rich", '--batch_size', str(batchsize), '--tracking.tracker', 'flow', datapth]

        #removing flow tracker
        if not tracker: call = ['sleap-track', '--model', model, '-o', outfl,
                 "--verbosity", "rich", '--batch_size', str(batchsize), datapth]
        
    # add in cpu call, sometimes for mac (with integrated hardware this needs to be cpu and not gpu)
    if not gpu: call.insert(1, '--cpu')
        
    call = ' '.join(call)

    #c = 'bash -c "conda activate sleap; '+call+'"'
    c = 'conda run -n {} '.format(enviroment_name)+call #this allows you run sleap through the sleap anaconda environment
    if givecall: print(c)
    sp.run(c, shell=True)
    return outfl

def SLEAP_convert(input_pth, datapth, outtype='analysis', enviroment_name='sleap', batchsize=8, givecall=False):
    #sleap-convert --format h5 E:/dynamic_xray/sleap_output/No0005-Original_PA_Deep.avi.predictions.slp
    #outtype='analysis'#best don't use others: 'json','h5'
    if not datapth: call = ['sleap-convert', '--format', outtype, input_pth]
    if datapth: call = ['sleap-convert', '--format', outtype, '--video', datapth, input_pth]
    call = ' '.join(call)
    c = 'conda run -n {} '.format(enviroment_name)+call #this allows you run sleap through the sleap anaconda environment
    if givecall: print(c)
    sp.run(c, shell=True)
    return


def convert_to_pandas(filename, dst=False):
    '''
    '''

    fileprefix = os.path.basename(filename).replace('.avi.predictions.analysis.h5','')
    modelname = os.path.basename(os.path.dirname(filename))

    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        node_names = [n.decode() for n in f["node_names"][:]]
        tracks = f['tracks'][:].T #frames, nodes, (x,y), num of animals
        track_occupancy = f['track_occupancy'][:]
        track_names = f['track_names'][:]


    idx_nodes = {idx:nm for idx, nm in enumerate(node_names)}
    #generate dataframe of columns=labels, rows =frames
    cols = ['modelname', 'fileprefix', 'frame', 'label_num', 'label_id', 'x', 'y']
    data = []
    for frame,d in enumerate(tracks):
        for obj,dd in enumerate(d):
            data.append([modelname, fileprefix, frame, obj, idx_nodes[obj], dd[0][0] , dd[1][0]])

    df = pd.DataFrame(data=data, columns=cols)

    #convert dtypes
    for col in ['x','y']:
        df[col] = df[col].astype('float32')

    #now cal velocity and others
    vel, body_part_treated_as_center = calc_velocity(df.copy())

    try:
        time_freqzscore = calc_time_freqzscore(df, vel.copy(), body_part_treated_as_center)
        if dst: np.save(dst+'/time_freqzscore_{}.npy'.format(fileprefix), time_freqzscore)
    except:
        with open(dst+'/error_in_time_freqscore.txt', 'w+') as txtfl:
            txtfl.write('Error'); txtfl.close()

    df = pd.merge(df, vel)
    if dst: df.to_csv(dst+'/dataframe_{}.csv'.format(fileprefix))

    return dst+'/dataframe_{}.csv'.format(fileprefix)


def calc_velocity(df):

    #build rolling average
    vel_interval = 3 #number of prior x's to sample and future
    ego_centric_interval = 0 #number prior and after to average over, i.e 1 == over 3 total frames
    cols = ['label_id', 'frame','velocity', 'xchange', 'ychange', 'vel_rolling', 'distance_from_start',  'distance_from_med', 'xvel_rolling', 'yvel_rolling', 'rolling_xmed', 'rolling_ymed', 'rolling_centered_x', 'rolling_centered_y']
    df = df.sort_values('label_id')
    dflst=[]

    for frame in range(0,df['frame'].max()):
        #first frame is zero for changes/velocity
        if frame == 0:
            lab_id_len = len(df[df['frame']==frame]['label_id'])
            zero_arr = np.zeros((lab_id_len))
            a = np.asarray([df[df['frame']==frame]['label_id'].values, zero_arr]+ [zero_arr]*(len(cols)-2)).T

        else:
            #velocity and x/ychanges
            f0 = df[df['frame']==frame-1].interpolate(); f0.sort_values('label_id')
            f1 = df[df['frame']==frame].interpolate(); f1.sort_values('label_id')
            vel=np.sqrt(((f1[['x','y']].values - f0[['x','y']].values)**2).sum(1)) #this assumes that frame rate is constant (not dividing by time)
            xchange, ychange = np.asarray(f1[['x','y']].values - f0[['x','y']].values).T

            #rolling
            f0_int = df[(df['frame']<=frame) & (df['frame']>=frame-vel_interval)].interpolate().groupby('label_id').mean(); f0_int.sort_values('label_id')
            f1_int = df[(df['frame']>=frame)&(df['frame']<=frame+vel_interval)].interpolate().groupby('label_id').mean(); f1.sort_values('label_id')
            vel_rolling=np.sqrt(((f1_int[['x','y']].values - f0_int[['x','y']].values)**2).sum(1))

            #delta from starting point
            f0 = df[df['frame']==0].interpolate(); f0.sort_values('label_id')
            f1 = df[df['frame']==frame].interpolate(); f1.sort_values('label_id')
            distance_from_start=np.sqrt(((f1[['x','y']].values - f0[['x','y']].values)**2).sum(1))

            #delta from median
            xmed_across_frames, ymed_across_frames = pd.DataFrame(df.groupby('label_id').median().sort_values('label_id'))[['x','y']].values.T
            distance_from_med=np.sqrt(((f1[['x','y']].values - np.asarray([xmed_across_frames, ymed_across_frames]).T)**2).sum(1))

            #x/y vel rolling
            xvel_rolling, yvel_rolling=(f1_int[['x','y']].values - f0_int[['x','y']].values).T

            #variable that looks at difference from median value and have it be +/- in y
            rolling_ymed = ymed_across_frames - f1_int['y'].values #order is important here
            rolling_xmed = f1_int['x'].values - xmed_across_frames #order is important here

            #make ego centric coordinates by subtracting rolling mean of center from part (down and right are positive in keep w convention of SLEAP)
            f1_ego = df[(df['frame']<=frame+ego_centric_interval) & (df['frame']>=frame-ego_centric_interval)].interpolate().groupby('label_id').mean()
            body_part_treated_as_center=list(set(f1_ego.index.tolist()) & set(['S AK', 'RU AArch', 'RU RV', 'LV LU']))[0]
            rolling_centered_x, rolling_centered_y = (f1_ego[['x','y']].values - f1_ego[f1_ego.index==body_part_treated_as_center][['x','y']].values).astype('float16').T

            #add to list
            a=np.asarray([f1['label_id'].values, [int(frame)]*len(f1['label_id'].values), vel, xchange, ychange, vel_rolling, distance_from_start, distance_from_med, xvel_rolling, yvel_rolling, rolling_xmed, rolling_ymed, rolling_centered_x, rolling_centered_y]).T

        dflst.append(pd.DataFrame(data=a, columns=cols))

    ndf =  pd.concat(dflst)
    ndf['frame'] = ndf['frame'].astype('int')
    ndf['rolling_centered_y'] = ndf['rolling_centered_y'].astype('float32')
    ndf['rolling_centered_x'] = ndf['rolling_centered_x'].astype('float32')

    precision = 'float16'
    for c in cols[2:]:
        ndf[c] = ndf[c].astype(precision)
    ndf = ndf.sort_values('frame', ascending=True)
    del dflst, df; gc.collect()

    return ndf, body_part_treated_as_center

def calc_time_freqzscore(df, ndf, body_part_treated_as_center):

    #build rolling average
    vel_interval = 3 #number of prior x's to sample and future
    ego_centric_interval = 0 #number prior and after to average over, i.e 1 == over 3 total frames
    cols = ['label_id', 'frame','velocity', 'xchange', 'ychange', 'vel_rolling', 'distance_from_start',  'distance_from_med', 'xvel_rolling', 'yvel_rolling', 'rolling_xmed', 'rolling_ymed', 'rolling_centered_x', 'rolling_centered_y']
    df = df.sort_values('label_id')
    dflst=[]

    time_freqzscore = []
    #now zscore across centered coordinates
    for val in ['rolling_centered_x', 'rolling_centered_y']:
        #szcore to keep
        zdf = ndf.pivot_table(values = val, index = 'label_id', columns = 'frame')[range(len(ndf.frame.unique()))]
        zsc= scipy.stats.zscore(zdf, axis=1)
        zdf = pd.DataFrame(zsc, columns=zdf.columns, index=zdf.index)
        zdf = zdf.reset_index().melt('label_id', var_name='frame', value_name=val+'_zscore')
        ndf=ndf.merge(zdf)

        #####RIGHT NOW YOU ARE dyadically spacing things, just linearly
        #NOW cwt; freqs = frequency x part x frame
        zdf = ndf.pivot_table(values = val, index = 'label_id', columns = 'frame')[range(len(ndf.frame.unique()))]
        zsc= scipy.stats.zscore(zdf[zdf.index!=body_part_treated_as_center], axis=1) #this time drop centered part
        coef, freqs=pywt.cwt(zsc, scales = np.arange(1,100,4), wavelet = 'morl',axis=1) #adjusting the scales greatly changes things (plt.matshow(coef))

        #concatenate into single vector 2*(#parts - 1)*#freqs [2 for x&y, freqs should be 25; minus 1 for centered body part]
        #this is now time x (freq*parts)
        time_freqzscore.append(coef.reshape(coef.shape[0]*coef.shape[1], coef.shape[2]))

    #now reshape/ravel x with y and save out
    time_freqzscore = np.asarray(time_freqzscore)
    time_freqzscore = time_freqzscore.reshape((time_freqzscore.shape[0]*time_freqzscore.shape[1], time_freqzscore.shape[2]))

    gc.collect()

    return time_freqzscore



def build_ethogram(mp4name, csv_file, structures_to_include, category_to_plot, dst, fps):

    #dst
    makedir(dst)

    #setup tmpdst
    flnm = os.path.basename(mp4name).replace('.mp4','')
    tmpdst = dst+'/ethogram'; makedir(tmpdst)

    #load
    reader = imageio.get_reader(mp4name)
    df = pd.read_csv(csv_file)

    #populate array
    x,y = reader.get_meta_data()['source_size']
    data = np.zeros((reader.count_frames(), y,x))#,3))
    for i, im in enumerate(reader):
        data[i] = im[:,:,0]

    #get and mark points
    lnsp = np.linspace(0,1,len(df['label_id'].unique()))
    node_color_dct = {node:plt.get_cmap('rainbow')(lnsp[num]) for num,node in enumerate(df['label_id'].unique())}
    max_frames = df['frame'].max()
    vel = df[df['label_id'].isin(structures_to_include)].copy()

    #build min/max ylim dct
    ylimdct = {struc:[df[df['label_id']==struc][category_to_plot].min(), df[df['label_id']==struc][category_to_plot].max()] for struc in structures_to_include}

    for i,im in enumerate(data):
        try:
            tvel = vel[vel['frame']<=i]
            #make image
            ax1 = plt.subplot(121); ax1.margins(0.05)
            ax1.imshow(im, cmap='Greys'); ax1.axis('off')
            for ii, r in df[df['frame']==i][['x','y','label_id']].dropna().iterrows(): #drop na gets rid of points SLEAP didn't find
                ax1.text(r['x']+3, r['y']-3, r['label_id'], size=4, color=node_color_dct[r['label_id']])
                ax1.scatter([r['x']], [r['y']], s=1, color=node_color_dct[r['label_id']])
            #plot
            for iii, struc in enumerate(structures_to_include):
                ax2 = plt.subplot(len(structures_to_include),2,(iii+1)*2)
                ax2.margins(2, 2)
                x,y=tvel[tvel['label_id']==struc][['frame',category_to_plot]].values.T
                ax2.plot(x,y, color=node_color_dct[struc])
                ax2.text(max_frames,5,struc, color=node_color_dct[struc])
                ax2.set_xlim(0,max_frames)
                ax2.set_ylim(ylimdct[struc][0],ylimdct[struc][1]) #*1.05
                ax2.axis('off')

            plt.savefig(tmpdst+'/im_{}.png'.format(str(i).zfill(5)), dpi=300); plt.close()
        except:
            print(i)

    #now save out as avi and delete folder
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warn_fxn()

        fls = listall(tmpdst); fls.sort()
        writer = imageio.get_writer(tmpdst+'.mp4', fps=fps)

        for i,fl in enumerate(fls):
            #w.append_data(imageio.imread(fl))
            writer.append_data(imageio.imread(fl))
        writer.close()

    shutil.rmtree(tmpdst)
    del data; gc.collect()

    return

    #for below
    dt = datetime.datetime.today()
    #calculate area
def calculate_area(dfpth):
        '''
        Function to calculate area given a dataframe to load

        list_of_structures = []

        dfpth = 'E:/dynamic_xray/sleap_outputs_japenesedata/dataframes/dataframe_No0005-Original_PA_Deep.mp4.predictions.analysis.h5.csv'
        dfpth = 'E:/dynamic_xray/sleap_outputs_japenesedata/dataframes/dataframe_No0005-Original_PA_Natural.mp4.predictions.analysis.h5.csv'

        for substructures in [llung_lheart, rlung_rheart]: #rlung, llung, heart, total_lung
            arr = df[(df['frame']==z) & (df['label_id'].isin(substructures))][['x','y']].dropna().values
            plt.scatter(arr[:,0], -arr[:,1])

        sns.lineplot(data=ndf, x='frame', y='area', hue='structure')
        sns.lineplot(data=ndf[ndf['structure'].isin(['lungs', 'lungs-heart'])], x='frame', y='area', hue='structure')

        '''
        structures = {'llung':llung, 'rlung':rlung, 'heart':heart, 'llung_lheart':llung_lheart, 'rlung_rheart':rlung_rheart, 'total_lung':total_lung}
        rdiaphragm = ['R Lat D', 'R RU D', 'R Cen D', 'R LU D', 'R Med D'] #lateral to medial
        ldiaphragm = ['L Lat D', 'L LU D', 'L Cen D', 'L RU D', 'L Med D'] #lateral to medial


        #load
        df = pd.read_csv(dfpth)

        #get points and calculate area
        ndata = []
        for z in df['frame'].unique():

            for structure,substructures in structures.items():
                #get x,y locations for substructures; need to get rid of nans as cv2 contour break
                arr = df[(df['frame']==z) & (df['label_id'].isin(substructures))].set_index('label_id').reindex(substructures)[['x','y']].dropna().values #needs to be done this way to ensure appropriate order for convex hull
                area = cv2.contourArea(arr.astype(int))
                ndata.append([structure, z, np.int(area)])

                #lungs minus heart
                if structure == 'total_lung': lungs_area = area
                if structure == 'heart': heart_area = area

            ndata.append(['lungs-heart', z, lungs_area - heart_area])

            #now calculate curvature index of each diaphragm (radius vertical / radius horizontal)
            #will have large negative number if cannot find one of the points
            for i, diaphragm in enumerate([ldiaphragm, rdiaphragm]):
                tdf = df[df['frame']==z]


                ''' OLD WAY, this failed often when point tracking missed one of the points even for a single frame
                latd = tdf[tdf['label_id']==diaphragm[0]][['x','y']].values[0]
                medd = tdf[tdf['label_id']==diaphragm[-1]][['x','y']].values[0]
                cend = tdf[tdf['label_id']==diaphragm[2]][['x','y']].values[0]
                '''
                #NEW WAY
                #to take care of nans do the following: get structures, order them, take most medial/lateral non nan.
                #Then find middle point or average of two middle points if even number of points
                ttdf = tdf[tdf['label_id'].isin(diaphragm)]
                #this allows you to sort (lat to medial) and then drop nas
                ttdf = ttdf.set_index('label_id')
                ttdf = ttdf.loc[diaphragm][['x','y']].dropna()
                latd = ttdf.values[0]
                medd = ttdf.values[-1]
                if ttdf.empty:
                    #dummy values so you know net failed here
                    rh = -100
                    rv = -100
                    cindex = -100
                else:
                    #even / odd
                    if ttdf.shape[0] % 2 == 1: #odd
                        cend = ttdf.values[int(ttdf.shape[0]/2)]
                    else: # even; so average
                        middle = int(ttdf.shape[0] / 2)
                        cend = ttdf.values[middle-1:middle+1].mean(0) #take averages

                    #then calculate things of interest
                    rh = point_distance(latd, medd).astype('int')
                    rv = shortest_distance(latd, medd, cend).astype('int')
                    cindex = (rv/rh).astype('float32')
                diaphragm_label = ['l_diaphragm', 'r_diaphragm'][i]
                ndata.append(['{}_rad_horiz'.format(diaphragm_label), z, rh])
                ndata.append(['{}_rad_vert'.format(diaphragm_label), z, rv])
                ndata.append(['{}_curvature_index'.format(diaphragm_label), z, cindex])

        ndf = pd.DataFrame(data=ndata, columns = ['structure', 'frame', 'area'])
        ndf['area'] = ndf['area'].astype('float64')
        ndf = ndf.pivot_table(values = 'area', index = 'frame', columns = 'structure').reset_index()
        return ndf


def calculate_area_with_videos(dfpth, vidpth, svfld, plot_areas=True, alpha=0.5, fps=12, structures_to_show=['llung_lheart', 'rlung_rheart', 'heart']):
        '''
        Function to calculate area given a dataframe to load

        list_of_structures = []

        dfpth = 'E:/dynamic_xray/sleap_outputs_japenesedata/dataframes/dataframe_No0005-Original_PA_Deep.mp4.predictions.analysis.h5.csv'
        vidpth = 'E:/dynamic_xray/data/JapeneseData_mp4/No0005-Original_PA_Deep.mp4'
        svfld = 'C:/Users/tpisano/Downloads'

        for substructures in [llung_lheart, rlung_rheart]: #rlung, llung, heart, total_lung
            arr = df[(df['frame']==z) & (df['label_id'].isin(substructures))][['x','y']].dropna().values
            plt.scatter(arr[:,0], -arr[:,1])

        '''
        structures = {'llung':llung, 'rlung':rlung, 'heart':heart, 'llung_lheart':llung_lheart, 'rlung_rheart':rlung_rheart, 'total_lung':total_lung}
        rdiaphragm = ['R Lat D', 'R RU D', 'R Cen D', 'R LU D', 'R Med D'] #lateral to medial
        ldiaphragm = ['L Lat D', 'L LU D', 'L Cen D', 'L RU D', 'L Med D'] #lateral to medial

        #visualization
        #structures_to_show = ['llung_lheart', 'rlung_rheart', 'heart']#, 'r_diaphragm_curvature_index', 'l_diaphragm_curvature_index']
        structures_to_show_names = {'llung_lheart':'Left Lung', 'rlung_rheart':'Right Lung', 'heart':'Heart', 'total_lung':'Total Lung'}#, 'r_diaphragm_curvature_index': 'Right Curvature index', 'l_diaphragm_curvature_index': 'Left Curvature index'}


        #load
        df = pd.read_csv(dfpth)
        reader = imageio.get_reader(vidpth)
        vid = np.asarray([im[:,:,:] for i, im in enumerate(reader)])

        #colors
        lnsp = np.linspace(0,1,len(structures_to_show))
        node_color_dct = {node:plt.get_cmap('rainbow')(lnsp[num]) for num,node in enumerate(structures_to_show)}

        #get points and calculate area of total lung, minus heart, and llung and right lung
        ndata = []; nvid = np.zeros_like(vid)
        for z in df['frame'].unique():

            #for each frame generate an image
            im = np.invert(np.copy(vid[z]))

            for structure,substructures in structures.items():

                #get x,y locations for substructures; needs to be in order; need to get rid of nans as cv2 contour break
                arr = df[(df['frame']==z) & (df['label_id'].isin(substructures))].set_index('label_id').reindex(substructures)[['x','y']].dropna().values #needs to be done this way to ensure appropriate order for convex hull
                area = cv2.contourArea(arr.astype(int))

                #modify image
                if structure in structures_to_show:

                    #plot
                    tim = np.zeros_like(im)
                    cv2.fillPoly(tim, pts = [arr.astype('int32')], color = tuple(xx*255 for xx in node_color_dct[structure]))
                    mask = tim.astype(bool)
                    im[mask] = cv2.addWeighted(im, alpha, tim, 1 - alpha, 0)[mask]

                #now calculate area
                ndata.append([structure, z, np.int(area)])

                #lungs minus heart
                if structure == 'total_lung': lungs_area = area
                if structure == 'heart': heart_area = area

            ndata.append(['lungs-heart', z, lungs_area - heart_area])

            #now calculate curvature index of each diaphragm (radius vertical / radius horizontal)
            #will have large negative number if cannot find one of the points
            for i, diaphragm in enumerate([ldiaphragm, rdiaphragm]):
                tdf = df[df['frame']==z]
                latd = tdf[tdf['label_id']==diaphragm[0]][['x','y']].values[0]
                medd = tdf[tdf['label_id']==diaphragm[-1]][['x','y']].values[0]
                cend = tdf[tdf['label_id']==diaphragm[2]][['x','y']].values[0]
                rh = point_distance(latd, medd).astype('int')
                rv = shortest_distance(latd, medd, cend).astype('int')
                cindex = (rv/rh).astype('float32')
                diaphragm_label = ['l_diaphragm', 'r_diaphragm'][i]
                ndata.append(['{}_rad_horiz'.format(diaphragm_label), z, rh])
                ndata.append(['{}_rad_vert'.format(diaphragm_label), z, rv])
                ndata.append(['{}_curvature_index'.format(diaphragm_label), z, cindex])

            #plt.matshow(im)
            nvid[z,:,:,:] = im

        #reshape data
        ndf = pd.DataFrame(data=ndata, columns = ['structure', 'frame', 'area'])
        ndf['area'] = ndf['area'].astype('float64')
        ndf = ndf.pivot_table(values = 'area', index = 'frame', columns = 'structure').reset_index()
        ndf.to_csv(os.path.join(svfld, os.path.basename(vidpth).replace('avi','').replace('.mp4', '') + '_areas_and_curvature_indices.csv'), index=False)

        if plot_areas == False:
            #save out video
            writer = imageio.get_writer(os.path.join(svfld, os.path.basename(vidpth).replace('.mp4','') + '_areas.mp4'), fps=fps)
            for i,im in enumerate(nvid):
                writer.append_data(im)
            writer.close()
        else:
            max_frames = nvid.shape[0]
            ylims = {structure:[np.max((0, ndf[structure].min())), np.max((1,ndf[structure].max()))] for structure in structures_to_show} #have a floor of zero and one for min/max lims
            tmpdst = os.path.join(svfld, 'temporary_plots'); makedir(tmpdst)
            for i,im in enumerate(nvid):

                #make image
                ax1 = plt.subplot(121); ax1.margins(0.05)
                ax1.imshow(im, cmap='Greys_r'); ax1.axis('off')

                #plot
                for iii, structure in enumerate(structures_to_show):
                    ax2 = plt.subplot(len(structures_to_show),2,(iii+1)*2)
                    ax2.margins(2, 2)
                    #get
                    x,y = ndf[ndf['frame']<=i][['frame', structure]].dropna().values.T
                    ax2.plot(x,y, color=node_color_dct[structure])
                    ax2.text(max_frames-10,ylims[structure][0]*0.95,structures_to_show_names[structure], color=node_color_dct[structure], fontsize=8)
                    ax2.set_xlim(0,max_frames)
                    ax2.set_ylim(ylims[structure][0],ylims[structure][1])
                    ax2.axis('off')

                plt.savefig(tmpdst+'/im_{}.png'.format(str(i).zfill(5)), dpi=300); plt.close()

            #save out video
            fls = listall(tmpdst); fls.sort()
            writer = imageio.get_writer(os.path.join(svfld, os.path.basename(vidpth).replace('.mp4','') + '_areas.mp4'), fps=fps)
            for i,fl in enumerate(fls):
                writer.append_data(imageio.imread(fl))
            writer.close()
            shutil.rmtree(tmpdst)

        del ndata; gc.collect()
        return ndf


def plot_curvature_index(dfpth, area_dfpth, vidpth, svfld, alpha=0.5, fps=12):
        '''

        list_of_structures = []
        dfpth = 'E:/dynamic_xray/sleap_outputs_japenesedata/dataframes/dataframe_No0005-Original_PA_Deep.mp4.predictions.analysis.h5.csv'
        area_dfpth = 'C:/Users/tpisano/Downloads\\No0005-Original_PA_Deep_areas_and_curvature_indices.csv'
        vidpth = 'E:/dynamic_xray/data/JapeneseData_mp4/No0005-Original_PA_Deep.mp4'
        svfld = 'C:/Users/tpisano/Downloads'
        '''
        rdiaphragm = ['R Lat D', 'R RU D', 'R Cen D', 'R LU D', 'R Med D'] #lateral to medial
        ldiaphragm = ['L Lat D', 'L LU D', 'L Cen D', 'L RU D', 'L Med D'] #lateral to medial
        structures = {'rdiaphragm':rdiaphragm, 'ldiaphragm':ldiaphragm}

        #visualization
        structures_to_show = ['r_diaphragm_curvature_index', 'l_diaphragm_curvature_index']
        structures_to_show_names = {'r_diaphragm_curvature_index': 'Right Curvature index', 'l_diaphragm_curvature_index': 'Left Curvature index'}
        crossmapdct = {'r_diaphragm_curvature_index': 'rdiaphragm', 'l_diaphragm_curvature_index': 'ldiaphragm'}

        #load
        df = pd.read_csv(dfpth)
        reader = imageio.get_reader(vidpth)
        vid = np.asarray([im[:,:,:] for i, im in enumerate(reader)])

        #colors
        lnsp = np.linspace(0,1,len(structures_to_show))
        node_color_dct = {node:plt.get_cmap('rainbow')(lnsp[num]) for num,node in enumerate(structures.keys())}

        #get points and calculate area of total lung, minus heart, and llung and right lung
        ndata = []; nvid = np.zeros_like(vid)
        for z in df['frame'].unique():

            #for each frame generate an image
            im = np.invert(np.copy(vid[z]))

            for structure,substructures in structures.items():

                #get x,y locations for substructures; needs to be in order; need to get rid of nans as cv2 contour break
                arr = df[(df['frame']==z) & (df['label_id'].isin(substructures))].set_index('label_id').reindex(substructures)[['x','y']].dropna().values.astype('int') #needs to be done this way to ensure appropriate order for convex hull

                #modify image
                tim = np.zeros_like(im)
                cv2.fillPoly(tim, pts = [arr.astype('int32')], color = tuple(xx*255 for xx in node_color_dct[structure]))
                mask = tim.astype(bool)
                im[mask] = cv2.addWeighted(im, alpha, tim, 1 - alpha, 0)[mask]

                #draw line
                for point1, point2 in zip(arr, arr[1:]):
                    cv2.line(im, tuple(point1), tuple(point2), tuple(xx*255 for xx in node_color_dct[structure]), 2)


            #plt.matshow(im)
            nvid[z,:,:,:] = im

        #reshape data
        ndf = pd.read_csv(area_dfpth)

        max_frames = nvid.shape[0]
        ylims = {structure:[np.max((0, ndf[structure].min())), np.max((1,ndf[structure].max()))] for structure in structures_to_show} #have a floor of zero and one for min/max lims
        tmpdst = os.path.join(svfld, 'temporary_plots'); makedir(tmpdst)
        for i,im in enumerate(nvid):

            #make image
            ax1 = plt.subplot(121); ax1.margins(0.05)
            ax1.imshow(im); ax1.axis('off')

            #plot
            for iii, structure in enumerate(structures_to_show):
                ax2 = plt.subplot(len(structures_to_show),2,(iii+1)*2)
                ax2.margins(2, 2)
                #get
                x,y = ndf[ndf['frame']<=i][['frame', structure]].dropna().values.T
                ax2.plot(x,y, color=node_color_dct[crossmapdct[structure]])
                ax2.text(max_frames-10,ylims[structure][0]-(0.05 * ylims[structure][0]),structures_to_show_names[structure], color=node_color_dct[crossmapdct[structure]], fontsize=8)
                ax2.set_xlim(0,max_frames)
                ax2.set_ylim(ylims[structure][0],ylims[structure][1])
                #sns.despine(top=False, right=True, left=True, bottom=False)
                ax2.axis('off')

            plt.savefig(tmpdst+'/im_{}.png'.format(str(i).zfill(5)), dpi=300); plt.close()

        #save out video
        fls = listall(tmpdst); fls.sort()
        writer = imageio.get_writer(os.path.join(svfld, os.path.basename(vidpth).replace('.mp4','') + '_curvature.mp4'), fps=fps)
        for i,fl in enumerate(fls):
            writer.append_data(imageio.imread(fl))
        writer.close()
        shutil.rmtree(tmpdst)

        gc.collect()
        return ndf

def make_distance_change_maps(svfld, dfpth, vidpth, cmap='coolwarm', vmin=-20, vmax=20, alpha=0.95, generate_distance_change_maps_videos=True):
    '''
    Function to make disance change maps somewhat like strain echocardiography

    '''

    #load
    df = pd.read_csv(dfpth)
    pntid = os.path.basename(os.path.dirname(os.path.dirname(dfpth)))
    structures = df.label_id.unique()

    #define partners - that is for each point, which two points to look at deltas for
    partners = {'Right Lung': rlung+['R Lat D'], 'Left Lung': ['L Lat D']+llung[::-1]}

    #now calculate distances between partners for each frame
    data = []
    for frame in range(df.frame.min(), df.frame.max()):
        for nm, lst in partners.items():
            for i, structure in enumerate(lst[:-1]):
                try:
                    edge = '{}_w_{}'.format(lst[i], lst[i+1])
                    n0 = df[(df.frame==frame)&(df.label_id==lst[i])][['x','y']].values[0]
                    n1 = df[(df.frame==frame)&(df.label_id==lst[i+1])][['x','y']].values[0]
                    data.append([lst[i], n0[0], n0[1], lst[i+1], n1[0], n1[1], edge, point_distance(n0, n1), frame])
                except Exception: #deals with missing points
                    edge = '{}_w_{}'.format(lst[i], lst[i+1])
                    data.append([lst[i], np.nan, np.nan, lst[i+1], np.nan, np.nan, edge, np.nan, frame])
    ndf = pd.DataFrame(data, columns = ['node0', 'node0_x', 'node0_y', 'node1', 'node1_x', 'node1_y', 'edge', 'point_distance', 'frame'])
    ndf['patient_id'] = pntid
    ndf.to_csv(os.path.join(svfld, 'distance_change_between_points.csv'))

    #now plot % change from median
    means = ndf.groupby('edge').mean()['point_distance'].to_dict()

    ndf['percent_change_from_mean'] = ndf.apply(lambda xx:-100*np.round(1-xx['point_distance'] / means[xx['edge']],3), 1)

    nndf = ndf.pivot_table(index='edge', columns='frame', values='percent_change_from_mean')
    order = ['LU Med CW_w_L Med CW','L Med CW_w_LC CW','LC CW_w_L2','L2_w_L3','L3_w_L4','L4_w_L5','L5_w_L6','L6_w_L7','L7_w_L8','L8_w_L9','L9_w_L Lat D',
        'L Lat D_w_L LU D', 'L LU D_w_L Cen D', 'L Cen D_w_L RU D', 'L RU D_w_L Med D',
        'L Med D_w_LU Med CW', #consider deleting
        #RIGHT
        'RC CW_w_RU Med CW','RU Med CW_w_R Med CW',
        'R Med CW_w_R Med D', #consider deleting
        'R Med D_w_R LU D', 'R LU D_w_R Cen D','R Cen D_w_R RU D',
       'R RU D_w_R Lat D',
       'R Lat D_w_R9','R9_w_R8','R8_w_R7','R7_w_R6','R6_w_R5','R5_w_R4','R4_w_R3','R3_w_R2','R2_w_RC CW']
    nndf = nndf.T[order].T
    f, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(nndf, cmap=cmap, vmin=vmin, vmax=vmax, yticklabels=True, ax=ax, cbar_kws={'label': 'Percent change from mean'})
    ax.set_xlabel('Frame'); ax.set_ylabel('')
    plt.savefig(os.path.join(svfld, os.path.basename(vidpth).replace('.mp4','') + '_distance_changes.png'), dpi=300)#; plt.close()
    plt.savefig(os.path.join(svfld, os.path.basename(vidpth).replace('.mp4','') + '_distance_changes.pdf'), dpi=300); plt.close()

    if generate_distance_change_maps_videos:

        #now make images for video
        reader = imageio.get_reader(vidpth)
        vid = np.asarray([im[:,:,:] for i, im in enumerate(reader)])
        cmap0 = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        max_frames = vid.shape[0]
        tmpdst = os.path.join(svfld, 'temporary_plots'); makedir(tmpdst)

        #plot points on each image
        ndata = []; nvid = np.zeros_like(vid)
        for z in df['frame'].unique():
            #for each frame generate an image
            im = np.invert(np.copy(vid[z]))
            tdf = ndf[ndf.frame==z]
            x0,y0,x1,y1,mean_change = tdf[['node0_x', 'node0_y', 'node1_x', 'node1_y','percent_change_from_mean']].T.values
            line_colors = cmap0(norm(mean_change))
            #modify image
            tim = np.zeros_like(im)
            for i in range(tdf.shape[0]):
                tim = cv2.line(tim, (x0[i].astype('int'),y0[i].astype('int')), (x1[i].astype('int'),y1[i].astype('int')), 255*line_colors[i], thickness=8)
            mask = tim.astype(bool)
            im[mask] = cv2.addWeighted(im, alpha, tim, alpha, 0)[mask]
            nvid[z,:,:,:] = im

        for i,im in enumerate(nvid):
            #make image
            #ax1 = plt.subplot(121); ax1.margins(0.05)
            if False: #vertical version
                f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,12),gridspec_kw={'height_ratios': [1, 3]})
                ax1.imshow(im); ax1.axis('off')
                #plot 2
                sns.heatmap(nndf[::-1].T, cmap=cmap, vmin=vmin, vmax=vmax, yticklabels=False, xticklabels=False, ax=ax2, cbar_kws={'label': 'Percent change from mean'})
                ax2.annotate("", xy=(0, i), xytext=(-5, i),arrowprops=dict(arrowstyle="->"), fontsize=15)
                ax2.set_ylabel('Frame')
                ax2.set_xlabel('R Ribs, R Diaph, L Diaph, L Ribs')
                plt.tight_layout()

            if True:    #horizontal version
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,4),gridspec_kw={'width_ratios': [1, 5]})
                ax1.imshow(im); ax1.axis('off')
                #plot 2
                sns.heatmap(nndf, cmap=cmap, vmin=vmin, vmax=vmax, yticklabels=False, xticklabels=False, ax=ax2, cbar_kws={'label': 'Percent change from mean'})
                ax2.annotate("", xy=(i, 0), xytext=(i, -3),arrowprops=dict(arrowstyle="->"), fontsize=15)
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('R Ribs, R Diaphragm, L Diaphragm, L Ribs')
                plt.tight_layout()
            plt.savefig(tmpdst+'/im_{}.png'.format(str(i).zfill(5)), dpi=300); plt.close()

        #save out video
        fls = listall(tmpdst); fls.sort()
        writer = imageio.get_writer(os.path.join(svfld, os.path.basename(vidpth).replace('.mp4','') + '_distance_changes.mp4'), fps=fps)
        for i,fl in enumerate(fls):
            writer.append_data(imageio.imread(fl))
        writer.close()
        shutil.rmtree(tmpdst)
        gc.collect()
    return

#%%

if __name__ == "__main__":
    # dPFT typically works with mp4s. There are ways to convert from dicom or AVI to mp4.
    # depending on your data/machine, there might be conversion issues between data type, ensure there is sufficient contrast
    # prior to using CLAHE as conversions and improper data types can signficantly affect performance
    # some compensation for this can be done using CLAHE, but it is best to ensure no loss of resolution prior
    # most versions of Konica Minolta software have the ability to output avis

    #location of data, for this example the dPFT github folder
    dxr_fld = 'E:/dPFT'#'/Users/tjp7rr1/Python/dPFT/' #set your path here to the dPFT folder
    
    # load the image using imageio
    arr = np.asarray(imageio.mimread(os.path.join(dxr_fld, 'data/1_PA_original.mp4'), memtest=False))[:,:,:,0] #grayscale image, only need one channel

    #lets take a look at the image
    plt.imshow(arr[10], cmap='Greys_r')

    #You can see the lung but the contrast isn't great. Let's apply the CLAHE filter as a preprocessing step
    mp4_clahe = os.path.join(dxr_fld, 'data/1_PA_clahe.mp4') #this is the output mp4 file after CLAHE preprocessing

    # now apply filter and save
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(10,10)) # clip limit and tilegridsizes emperically work well for PA images
    narr = np.zeros_like(arr)
    for i, img in enumerate(arr):
        narr[i] = clahe.apply(img)
    imageio.mimwrite(mp4_clahe, np.asarray(narr), fps = 12, **{'macro_block_size':16})

    #Again lets take a look and you'll see much improved contrast
    plt.imshow(narr[10], cmap='Greys_r')

    # now lets run the inference
    # you will need to make sure you have downloaded the CNNs for SLEAP processing
    # downlaod: https://drive.google.com/drive/folders/19RFi1TA63rSfXhV1YV_W1SF-E4Ch2J5e?usp=drive_link
    
    #set paths to the CNNs
    cnn_fld = 'Z:/dxr_public'#'/Users/tjp7rr1/Downloads/dxr_public/'
    
    # Name of anaconda enviroment sleap is installed under, this typically should be different than dPFT conda environment
    enviroment_name = 'sleap' 
    
    # order must be 'first_centroid' then 'second_centered' for each lung field
    models = {'left_diaphragm' : [os.path.join(cnn_fld, 'left_lower_lung_first_centroid'), os.path.join(cnn_fld, 'left_lower_lung_second_centered')],
              'right_diaphragm' : [os.path.join(cnn_fld, 'right_lower_lung_first_centroid'), os.path.join(cnn_fld, 'right_lower_lung_second_centered')],
              'lulungs': [os.path.join(cnn_fld, 'left_upper_lung_first_centroid'), os.path.join(cnn_fld, 'left_upper_lung_second_centered')],
              'rulung': [os.path.join(cnn_fld, 'right_upper_lung_first_centroid'), os.path.join(cnn_fld, 'right_upper_lung_second_centered')]}

    #file to process - this should be a clahe mp4 file
    mp4_clahe = os.path.join(dxr_fld, 'data/1_PA_clahe.mp4')

    # set destination and make nested folders
    dst = 'C:/Users/tpisano/Downloads/dxr_tmp'#'/Users/tjp7rr1/Downloads/cnn_output'
    inference_dst = os.path.join(dst, 'inference')
    csv_file = mp4_clahe.replace('clahe.mp4', '.predictions.analysis.h5.csv')
    combineddfpth = os.path.join(inference_dst, 'combined_dataframe.csv')
    makedir(dst); makedir(inference_dst)
    
    ##switches - adjust these based on desired output
    convert_to_dataframe = True
    batchsize = 1
    gpu = False # set this to true if asking sleap to run on GPU, note that with Mac M1 pro, somethings you need to use "CPU" mode (thus gpu=False)
    provide_training_json=True #if sleap is not outputting any predictions this should be made to true (this occurs w different versions of sleap)
    
    #run inference
    for modelnm, model in models.items():
        
        if provide_training_json:
            for i,m in enumerate(model):
                model[i] = listdirfull(m, keyword='training_config.json')[0]
        
        
        try:
            minffld = os.path.join(inference_dst, modelnm); makedir(minffld)
            #run inference and convert to analysis file
            input_pth = os.path.join(minffld, '{}.predictions.slp'.format(os.path.basename(mp4_clahe)))
            input_pth = SLEAP_inference(model, mp4_clahe, tracker=True, dst=minffld, enviroment_name=enviroment_name, batchsize=batchsize, givecall=True, gpu=gpu)
            SLEAP_convert(input_pth, mp4_clahe, outtype='analysis', enviroment_name=enviroment_name, batchsize=batchsize, givecall=True)
            if convert_to_dataframe: df = convert_to_pandas(filename = input_pth.replace('.slp', '.analysis.h5'), dst=minffld)
        except:
            print('Failed {}'.format(modelnm))

        #Combine outputs into single csv
        df = pd.concat([pd.read_csv(xx) for xx in listall(inference_dst, keyword='.csv')])
        #remove structures that were picked up by multiple nets
        df = df.drop_duplicates(subset=['frame', 'label_id'], keep='last') #df = df.sort_values(by=['frame','label_id'])
        #remove any columns with unnamed in it, this is important.
        df = df.loc[:,~df.columns.str.startswith('Unnamed')]

        df.to_csv(combineddfpth, index=False)

    #now different ways to look at output
    fps=12
    dst_to_save_videos = os.path.join(dst, 'tracking_output_analysis'); makedir(dst_to_save_videos)

    #make ethogram vidoes
    ethogramylim = [-50, 50]
    category_to_plot = 'distance_from_med' # distance_from_med'#'distance_from_med'#'distance_from_start' #'vel_rolling'#'velocity'#'ychange' #velocity, xchange, y
    structures_to_include = ['L7', 'R7', 'AtAp LV', 'Lat LV', 'LL LV', 'L Cen D', 'R Cen D',]#'L4', 'L6', 'R4', 'R6' #'LC CW', 'RC CW','L Lat D', 'R Lat D','S AK', 'L AK', 'L Med D', 'R Med D', 'R LU D', 'R RU D', 'L LU D', 'L RU D', 'L7', 'R7','Lat RV','R Coracoid', 'L Coracoid', 'L SternClav', 'R SternClav', 'R Prox Clav', 'L Prox Clav', 'R Mid Clav', 'L Mid Clav', 'R Dist Clav', 'L Dist Clav'
    subtract_value=0.0 #between 0 and 1, use for preprocessing
    build_ethogram(mp4_clahe, csv_file=combineddfpth, structures_to_include=structures_to_include, category_to_plot=category_to_plot, dst = dst_to_save_videos, fps=fps)

    #AREA with video 
    structures_to_show = ['llung_lheart', 'rlung_rheart', 'heart']#, 'r_diaphragm_curvature_index', 'l_diaphragm_curvature_index'], 'total_lung'
    calculate_area_with_videos(dfpth=combineddfpth, vidpth=mp4_clahe, svfld=dst_to_save_videos, plot_areas=True, alpha=0.5, fps=fps, structures_to_show=structures_to_show)
    
    #calculate area by itself (without videos as above)
    calculate_area(dfpth=combineddfpth)

    #curvature - breaks when CNN fails
    area_dfpth = listall(dst_to_save_videos, keyword='areas_and_curvature_indices')[0]
    _=plot_curvature_index(dfpth=combineddfpth, area_dfpth=area_dfpth, vidpth=mp4_clahe, svfld=dst_to_save_videos, alpha=0.5, fps=fps)

    #make distance change maps
    make_distance_change_maps(svfld=dst_to_save_videos, dfpth=combineddfpth, vidpth=mp4_clahe, generate_distance_change_maps_videos=generate_distance_change_maps_videos)








#%%
if __name__ == "__main__":

    lrfliplist= []#'287741', '779373', '905240', '8416138',
                 #'7441994', '6201720', '1846582', 'E320811', '7793735','3653292', '8373328', '8097919', '2632355'] #', '8373328', '8097919','2632355'#9052407, '2877412',


    ##switches
    inference = True #True
    convert_to_dataframe = True# True
    generate_mp4_vids = True
    generate_ethogram_vids = False
    generate_area_videos=False#True
    generate_curvature_index=False
    generate_distance_change_maps=True#False #this needs to be true to make videos (make both true for videos)
    generate_distance_change_maps_videos=False

    # Name of anaconda enviroment sleap is installed under
    enviroment_name = 'sleap' #'sleap_v1.1.4'

    ##settings
    fps=12 #output
    batchsize = 1
    cliplim = 8 #clahe clip limit, 8 suggested for PA
    tg = (10,10) #clahe tg, (10,10) suggested for PA
    #Ethogram
    ethogramylim = [-50, 50]
    category_to_plot = 'distance_from_med' # distance_from_med'#'distance_from_med'#'distance_from_start' #'vel_rolling'#'velocity'#'ychange' #velocity, xchange, y
    subtract_value=0.0 #between 0 and 1, use for preprocessing
    #categories to extract from dicoms
    cols = ['StudyDate', 'StudyTime', 'StudyDescription', 'SeriesDescription', 'PatientName', 'PatientID', 'PatientBirthDate', 'PatientAge', 'PatientSex', 'FrameTime', 'ExposureTime']
    #structures to include for point tracking videos
    structures_to_include = ['L7', 'R7', 'AtAp LV', 'Lat LV', 'LL LV', 'L Cen D', 'R Cen D',]#'L4', 'L6', 'R4', 'R6' #'LC CW', 'RC CW','L Lat D', 'R Lat D','S AK', 'L AK', 'L Med D', 'R Med D', 'R LU D', 'R RU D', 'L LU D', 'L RU D', 'L7', 'R7','Lat RV','R Coracoid', 'L Coracoid', 'L SternClav', 'R SternClav', 'R Prox Clav', 'L Prox Clav', 'R Mid Clav', 'L Mid Clav', 'R Dist Clav', 'L Dist Clav'

    # for calculate_area_with_videos
    structures_to_show = ['llung_lheart', 'rlung_rheart', 'heart']#, 'r_diaphragm_curvature_index', 'l_diaphragm_curvature_index'], 'total_lung'

    ##locations
    #input
    inputpth0 = 'E:/dynamic_xray/backup_data/Image'
    inputpth1 = 'E:/dynamic_xray/backup_data/data_from_konicaminolta'
    dicom_src = listall(inputpth0, keyword='.dcm') + listall(inputpth1, keyword='.dcm') #subfolders with dicoms

    #save loc
    bigdst = 'E:/dynamic_xray/processed'; makedir(bigdst)
    rundst = os.path.join(bigdst, '202204_data'); makedir(rundst)
    
    #temp for trouble shooting
    rundst = os.path.join(bigdst, '202404_data'); makedir(rundst)
    #dicom_src = [xx for xx in dicom_src if '1.2.392.200036.9107.307.17345.20180104' in xx]
    
    


    #model paths - if multiple centroid then centered
    #this is without scaling
    #models = {'left_diaphragm' : ["E:/dynamic_xray/sleap_segmented/ldiaphragm/models/220222_221838.centroid.n=1138", "E:/dynamic_xray/sleap_segmented/ldiaphragm/models/220223_071520.centered_instance.n=1138"],
    #          'right_diaphragm' : ["E:/dynamic_xray/sleap_segmented/rdiaphragm/models/220301_061713.centroid.n=913", "E:/dynamic_xray/sleap_segmented/rdiaphragm/models/220301_224121.centered_instance.n=913"],
    #          'lulungs': ['E:/dynamic_xray/sleap_segmented/upper_lungs/ap_lulungs/models/220209_112742.centroid.n=876', 'E:/dynamic_xray/sleap_segmented/upper_lungs/ap_lulungs/models/220209_214329.centered_instance.n=876'],
    #          'rulung': ['E:/dynamic_xray/sleap_segmented/upper_lungs/ap_rulungs/models/220207_100810.centroid.n=759', 'E:/dynamic_xray/sleap_segmented/upper_lungs/ap_rulungs/models/220303_062140.centered_instance.n=777']}

    ##with scaling
    #models = {'left_diaphragm' : ["E:/dynamic_xray/sleap_segmented/ldiaphragm/size_test/models/220307_210304.centroid.n=1138", "E:/dynamic_xray/sleap_segmented/ldiaphragm/models/220223_071520.centered_instance.n=1138"],
    #          'right_diaphragm' : ["E:/dynamic_xray/sleap_segmented/rdiaphragm/size_test/models/220307_142932.centroid.n=913", "E:/dynamic_xray/sleap_segmented/rdiaphragm/models/220301_224121.centered_instance.n=913"],
    #          'lulungs': ['E:/dynamic_xray/sleap_segmented/upper_lungs/ap_lulungs/size_test/models/220306_100344.centroid.n=876', 'E:/dynamic_xray/sleap_segmented/upper_lungs/ap_lulungs/models/220209_214329.centered_instance.n=876'],
    #          'rulung': ['E:/dynamic_xray/sleap_segmented/upper_lungs/ap_rulungs/size_test/models/220310_134845.centroid.n=777', 'E:/dynamic_xray/sleap_segmented/upper_lungs/ap_rulungs/models/220303_062140.centered_instance.n=777']}


    #with scaling
    models = {'left_diaphragm' : ["E:/dynamic_xray/sleap_segmented/ldiaphragm/size_test/models/220401_105108.centroid.n=1299", "E:/dynamic_xray/sleap_segmented/ldiaphragm/size_test/models/220401_194528.centered_instance.n=1299"],
              'right_diaphragm' : ["E:/dynamic_xray/sleap_segmented/rdiaphragm/size_test/models/220330_162029.centroid.n=1106","E:/dynamic_xray/sleap_segmented/rdiaphragm/size_test/models/220331_095322.centered_instance.n=1106"],
              'lulungs': ["E:/dynamic_xray/sleap_segmented/upper_lungs/ap_lulungs/size_test/models/220328_093913.centroid.n=918", "E:/dynamic_xray/sleap_segmented/upper_lungs/ap_lulungs/size_test/models/220328_141705.centered_instance.n=918"],
              'rulung': ['E:/dynamic_xray/sleap_segmented/upper_lungs/ap_rulungs/size_test/models/220310_134845.centroid.n=777', 'E:/dynamic_xray/sleap_segmented/upper_lungs/ap_rulungs/models/220303_062140.centered_instance.n=777']}


    if generate_mp4_vids:
        ##run coversion loop
        for iii, dicompth in list(enumerate(dicom_src))[1:]:
            #if iii == 1:break
            print(r'\n\n{} of {}\n          {}'.format(iii, len(dicom_src), dicompth))
            #load
            ds = dcmread(dicompth)
            try:
                age = ds.PatientAge
            except:
                age = 'unknown'
            try:
                #dicom extraction ###SET UP AP VS RL
                #arr = dicom2arr(dicompth, subtract_value=subtract_value)
                arr = np.invert(pydicom.dcmread(dicompth).pixel_array)
                arr = cv2.convertScaleAbs(arr, alpha=(255.0/65535.0))
                #arr = skimage.img_as_ubyte(arr)
                
                if len(arr.shape) < 3:
                    print('   Not 3D array skipping')
                else:
                    try:
                        #build out new save structures and dataframe
                        data = {ds[c].name:str(ds[c].value) for c in cols}
                        ap_rl_suffix = 'PA' if not data['Series Description'] == 'Erect LAT' else 'RL'
                        flnm = 'PatientID_{}_Age_{}_DDRDate_{}_{}'.format(ds.PatientID, age, ds.ContentDate, ap_rl_suffix)
                        print(r'\n    {}\n\n\n'.format(flnm))
                        flnmfld = os.path.join(rundst, flnm); makedir(flnmfld)

                        #so not to redo work
                        if len(listall(flnmfld, keyword='_PA_clahe.mp4'))==0:

                            data['original filename'] = flnm; data['Study ID'] = str(ds.StudyID)
                            df = pd.DataFrame([data])
                            df['folder name'] = os.path.basename(flnmfld)
                            mp4name=os.path.join(flnmfld, '{}_{}_clahe.mp4'.format(df['Patient ID'].values[0], ap_rl_suffix))
                            df['mp4filename'] = mp4name
                            df.to_csv(os.path.join(flnmfld, 'dataframe.csv'), index=False)

                            #flip, LR, init clahe and populate array and save
                            arr = np.flip(arr, 2) #all need to be flipped
                            if np.any([xx in flnm for xx in lrfliplist]): arr = np.flip(arr, 2) #only a few do not
                            imageio.mimwrite(mp4name.replace('clahe','original'), arr, 'FFMPEG', fps = fps, **{'macro_block_size':16}) #original
                            clahe = cv2.createCLAHE(clipLimit=cliplim, tileGridSize=tg)
                            narr = np.zeros_like(arr)
                            for i, img in enumerate(arr):
                                narr[i] = clahe.apply(img)
                            imageio.mimwrite(mp4name, narr, fps = fps, **{'macro_block_size':16})

                            if False:
                                if flnm in lrfliplist: #flip LR
                                    print('flipping LR')
                                    vid = imageio.get_reader(mp4name)
                                    arr = np.asarray([i[1] for i in enumerate(vid)])
                                    arr = np.flip(arr, 2) ##flip and save it
                                    imageio.mimwrite(mp4name, arr, fps = vid.get_meta_data()['fps'], **{'macro_block_size':16})
                    except:
                        print('ERROR {}'.format(dicompth))
            except:
                print('ERROR {}'.format(dicompth))


       ##inference on AP only
    errors=[]
    flnmflds = listdirfull(rundst, '_PA')
    for iii, flnmfld in list(enumerate(flnmflds))[:]:
        print(r'\n\n{} of {}\n          {}'.format(iii, len(flnmflds), flnmfld))
        try:
             #setup save paths
             mp4name = listall(flnmfld,keyword='clahe.mp4')[0]
             inffld = os.path.join(flnmfld, 'inference')
             csv_file = mp4name.replace('clahe.mp4', '.predictions.analysis.h5.csv')
             combineddfpth = os.path.join(inffld, 'combined_dataframe.csv')
             #so not to redo work
             if len(listall(flnmfld, keyword='completed_inference_loop.txt'))==0:
                 makedir(inffld)

                 if inference:
                     for modelnm, model in models.items():
                         try:
                             minffld = os.path.join(inffld, modelnm); makedir(minffld)
                             #run inference and convert to analysis file
                             input_pth = os.path.join(minffld, '{}.predictions.slp'.format(os.path.basename(mp4name)))
                             if not np.any([True for xx in listall(minffld) if '.predictions.slp' in xx]): input_pth = SLEAP_inference(model, mp4name, tracker=True, dst=minffld, enviroment_name=enviroment_name, batchsize=batchsize, givecall=False)
                             if not np.any([True for xx in listall(minffld) if '.predictions.analysis.h5' in xx]): SLEAP_convert(input_pth, mp4name, outtype='analysis', enviroment_name=enviroment_name, batchsize=batchsize)
                             #input_pth = SLEAP_inference(model, mp4name, tracker=True, dst=minffld, batchsize=batchsize)
                             #SLEAP_convert(input_pth, mp4name, outtype='analysis', batchsize=batchsize)
                             if convert_to_dataframe: df = convert_to_pandas(filename = input_pth.replace('.slp', '.analysis.h5'), dst=minffld)
                         except:
                             print('Failed {}'.format(modelnm))

                 #Combine outputs into single csv
                 df = pd.concat([pd.read_csv(xx) for xx in listall(inffld, keyword='.csv')])
                 #remove structures that were picked up by multiple nets
                 df = df.drop_duplicates(subset=['frame', 'label_id'], keep='last') #df = df.sort_values(by=['frame','label_id'])
                 #remove any columns with unnamed in it, this is important.
                 df = df.loc[:,~df.columns.str.startswith('Unnamed')]

                 df.to_csv(combineddfpth, index=False)

                 #completed file
                 with open(flnmfld+'/completed_inference_loop.txt', 'w+') as txtfl:
                     txtfl.write('Completed'); txtfl.close()

             #make ethogram vidoes
             if generate_ethogram_vids and len(listall(flnmfld,keyword='ethogram.mp4'))==0:
                 build_ethogram(mp4name, csv_file=combineddfpth, structures_to_include=structures_to_include, category_to_plot=category_to_plot, dst = flnmfld, fps=fps)

             #AREA
             if generate_area_videos and len(listall(flnmfld,keyword='clahe_areas.mp4'))==0:
                 calculate_area_with_videos(dfpth=combineddfpth, vidpth=mp4name, svfld=flnmfld, plot_areas=True, alpha=0.5, fps=fps, structures_to_show=structures_to_show)
             else: calculate_area(dfpth=combineddfpth)

             #curvature - breaks when CNN fails
             if generate_curvature_index and len(listall(flnmfld,keyword='_curvature.mp4'))==0:
                 area_dfpth = listall(flnmfld, keyword='areas_and_curvature_indices')[0]
                 _=plot_curvature_index(dfpth=combineddfpth, area_dfpth=area_dfpth, vidpth=mp4name, svfld=flnmfld, alpha=0.5, fps=fps)

             #make distance change maps
             if generate_distance_change_maps and len(listall(flnmfld, keyword='PA_clahe_distance_changes.mp4'))==0:
                 make_distance_change_maps(svfld=flnmfld, dfpth=combineddfpth, vidpth=mp4name, generate_distance_change_maps_videos=generate_distance_change_maps_videos)

        except:
            print('Error in {}'.format(flnmfld))
            errors.append(flnmfld)

    #concatenate
    #dfs = [pd.read_csv(fl) for fl in listall(dfdst)]
    #df = pd.concat(dfs)
    #df.to_csv(dfdst+'concatenated_dataframe.csv')
if False:
    #copy over to google drive
    import shutil
    fld = 'E:/dynamic_xray/processed/202202_data'; makedir(fld)
    shutil.copytree(fld, fld+'gdrive')
    [os.remove(xx) for xx in listall(fld+'gdrive') if '.mp4' in xx or '.npy' in xx]
    gdrive = 'C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data'
    shutil.copytree(fld+'gdrive', os.path.join(gdrive, os.path.basename(fld)))

    #copy over area videos for screening
    src = "E:/dynamic_xray/processed/202203_data"
    fls = listall(src, keyword='_PA_clahe_areas.mp')
    dst = 'E:/dynamic_xray/processed/tmp/screening/size_test/fullsize'; makedir(dst)
    [shutil.copy(xx, dst) for xx in fls]

    src = "E:/dynamic_xray/processed/202203_data_size_test"
    fls = listall(src, keyword='_PA_clahe_areas.mp')
    dst = 'E:/dynamic_xray/processed/tmp/screening/size_test'; makedir(dst)
    [shutil.copy(xx, dst) for xx in fls]

    #copy over videos for added datasets
    src = "E:/dynamic_xray/processed/202203_iterative_screening/size_test/bad"
    fls = listall(src, keyword='_PA_clahe_areas.mp')
    nms = [os.path.basename(xx).replace('_areas.','.') for xx in fls]
    src1 = "E:/dynamic_xray/processed/202203_data_size_test"
    nfls = listall(src1, keyword='_PA_clahe.mp4')
    nfls = [xx for xx in nfls if os.path.basename(xx) in nms]
    dst = 'E:/dynamic_xray/processed/202203_iterative_screening/clahe_videos'; makedir(dst)
    [shutil.copy(xx, dst) for xx in nfls]

#%%#quickly append file names to files
if __name__=='__main__':
    #files for CNN dxr
    src = 'E:/dynamic_xray/processed/202202_data'; #remove_empty_folders(src)
    fls = listall(src, keyword='_PA_clahe_areas_and_curvature_indices')
    for fl in fls:
        df = pd.read_csv(fl)
        df['identifier'] = os.path.basename(fl).replace('_PA_clahe_areas_and_curvature_indices.csv', '')
        df.to_csv(fl)

    #%%
    #files for CNN dxr
    src = 'E:/dynamic_xray/processed/202202_data';#remove_empty_folders(src)
    fls = listall(src, keyword='_PA_clahe_areas_and_curvature_indices')
    cnndf = pd.concat([pd.read_csv(xx) for xx in fls])

    #Excel file
    pth = 'C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data/TP_WITH_EF_Unified_data 8.22.xlsx'
    pfdf = pd.read_excel(pth, engine='openpyxl', sheet_name='All TLC and RV Slides') #

    #intersection
    a = set([str(xx) for xx in cnndf['identifier'].unique()])
    b = set([str(xx) for xx in pfdf['Patient ID'].values])
    a.intersection(b)
    a.difference(b)
    #77

    #%%
    #%%
    #%%
    #ethogram
    fl = 'E:/dynamic_xray/sleap_output/dataframes/dataframe_No0005-Original_PA_Natural.csv'
    df = pd.read_csv(fl)
    parts=['Lat LV', 'L Cen D', 'R Cen D','L6', 'R6']
    tdf = df[(df['label_id'].isin(parts))]
    variable='rolling_ymed'#'ychange', 'yvel_rolling', 'rolling_ymed', 'xvel_rolling'
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(211)
    ax1.set_title('{}'.format(variable))
    sns.lineplot(data=tdf, x='frame', y=variable, hue='label_id',ax=ax1)
    ax1.set_xlim(0)
    sns.despine(top=True, left=True, right=True, bottom=True)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ttdf = tdf.pivot(index='label_id', columns='frame', values=variable)
    sns.heatmap(data=ttdf.T[parts].T,ax=ax2, cmap='RdBu', cbar=False, center=0)
    ax2.set_xlim(0)
    ax1.set_xticks([])
    ax1.legend(loc = 'upper right')
    ax1.set_xlabel('')
    plt.savefig('E:/dynamic_xray/sleap_output/ethograms/{}_{}_ethogram.jpeg'.format(os.path.basename(fl).replace('.csv',''), variable), dpi=300)


    #tracks colored by magnitude of speed

    def rand_jitter(arr):
        return arr + np.random.rand(*arr.shape) * 0.9

    fig = plt.figure()
    #ax1 = fig.add_subplot(121)
    #tdf = df[(df['label_id'].isin(['Lat RV', 'R Cen D']))]#&(df['frame']>=20)]
    #sns.lineplot(data=tdf, x='x', y='y',hue='label_id',ax=ax1)
    #ax1 = fig.add_subplot(122)
    #tdf = df[(df['label_id'].isin(['R Cen D']))]
    #sns.lineplot(data=tdf, x='x', y='y',ax=ax1)
    #ax1 = fig.add_subplot(121)
    ax1 = fig.add_subplot(111)
    tdf = df[(df['label_id'].isin(['Lat LV']))]
    #sns.lineplot(data=tdf, x='x', y='y',ax=ax1)
    xy = tdf[['x','y']].values.T
    x,y = rand_jitter(xy)
    t = tdf[['frame']].values.T
    sc=ax1.scatter(x,y,c=t,vmin=0, vmax=np.max(t),cmap='jet', alpha=0.9)
    ax1.plot(x,y,alpha=0.2)
    ax1.invert_yaxis()
    plt.colorbar(sc, label='Time')
    plt.title('Movement of Lat LV')
    plt.savefig('E:/dynamic_xray/sleap_output/ethograms/{}_rcd_movement.jpeg'.format(os.path.basename(fl).replace('.csv','')), dpi=300)
    #####
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    tdf = df[(df['label_id'].isin(['R Cen D']))]
    xy = tdf[['x','y']].values.T
    x,y = rand_jitter(xy)
    t = tdf[['frame']].values.T
    sc=ax1.scatter(x,y,c=t,vmin=0, vmax=np.max(t),cmap='jet', alpha=0.9)
    ax1.plot(x,y,alpha=0.2)
    ax1.invert_yaxis()
    plt.colorbar(sc, label='Time')
    plt.title('Movement R center diaphragm')
    plt.tight_layout()
    plt.savefig('E:/dynamic_xray/sleap_output/ethograms/{}_rcd_movement.jpeg'.format(os.path.basename(fl).replace('.csv','')), dpi=300)

    #%%
    #cluster
    fl = 'E:/dynamic_xray/sleap_output/dataframes/dataframe_No0005-Original_PA_Natural.csv'
    fl = 'E:/dynamic_xray/sleap_output/dataframes/dataframe_No0005-Original_PA_Deep.csv'
    df = pd.read_csv(fl)
    parts = ['R Coracoid','L Coracoid','L6', 'R6','LV LU','Lat LV', 'L Cen D','L Lat D', 'R Cen D','R Lat D']
    tdf = df[(df['label_id'].isin(parts))].copy()#&(df['frame']>=20)]

    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(211)
    ttdf = tdf.pivot(index='label_id', columns='frame', values='vel_rolling')
    ttdf = ttdf.T[parts].T.fillna(0)
    #sns.heatmap(data=ttdf,ax=ax1, cmap='viridis', cbar=False)
    #ax1.set_xlim(0)
    #ax1.set_xticks([])

    ax1.imshow(ttdf.values, aspect="auto", vmin=0, vmax=25, interpolation="nearest")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Nodes")
    ax1.set_yticks(np.arange(ttdf.shape[0]))
    ax1.set_yticklabels(ttdf.T.columns);
    ax1.set_xlim(0,ttdf.shape[1])
    ax1.set_xticks([])

    #cluster
    from sklearn.cluster import KMeans
    nstates = 3
    km = KMeans(n_clusters=nstates)

    labels = km.fit_predict(ttdf.values.T)
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.imshow(labels[None, :], aspect="auto", cmap="ocean_r", interpolation="nearest")
    ax2.set_xlabel("Frames")
    ax2.set_title("Ethogram (colors = clusters)");
    plt.savefig('E:/dynamic_xray/sleap_output/ethograms/{}_kmeans_cluster.jpeg'.format(os.path.basename(fl).replace('.csv','')), dpi=300)



    #ttdf = tdf.pivot(index='label_id', columns='frame', values='vel_rolling')
    #sns.heatmap(data=ttdf,ax=ax2, cmap='viridis', cbar=False)
    #plt.savefig('E:/dynamic_xray/sleap_output/ethograms/{}_ethogram.jpeg'.format(os.path.basename(fl).replace('.csv','')), dpi=300)

    #%%
    #all
    df = pd.read_csv('E:/dynamic_xray/sleap_output/dataframesconcatenated_dataframe.csv')
    del df['Unnamed: 0']
    del df['Unnamed: 0.1']

    y = df.groupby(['fileprefix', 'label_id']).median().astype('float32').fillna(0).copy()
    y = y[['x', 'y', 'velocity', 'xchange', 'ychange','vel_rolling', 'distance_from_start']]
    y = y.unstack()

    #tsne
    from sklearn.manifold import TSNE
    from matplotlib.ticker import NullFormatter
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(y)

    #COLOR NEXT TIME BASED ON AGE, SEX, ETCS...
    colors = [plt.cm.Spectral(xx) for xx in np.linspace(0,1,y.shape[0])]

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax1.scatter(Y[:, 0], Y[:, 1], c=colors, alpha=0.9)#, cmap=plt.cm.Spectral)
    ax1.set_title("tsne median")
    #plt.xaxis.set_major_formatter(NullFormatter())
    #plt.yaxis.set_major_formatter(NullFormatter())
    ax1.axis('tight')


    #THIS NEEDS TO HAVE X AND Y COORDINATES (AGE, COPD, PFTs, etc?)
    #kmeans
    from sklearn.cluster import KMeans
    nstates = 8
    km = KMeans(n_clusters=nstates)

    labels = km.fit_predict(y.values)
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.scatter(labels[:, 0], labels[:, 1], c=colors, alpha=0.9)
    ax2.set_xlabel("Frames")
    ax2.set_title("Ethogram (colors = clusters)");
    #plt.savefig('E:/dynamic_xray/sleap_output/ethograms/{}_kmeans_cluster.jpeg'.format(os.path.basename(fl).replace('.csv','')), dpi=300)

    #%% tsne
    import pickle
    from sklearn.manifold import TSNE
    #from openTSNE import TSNE
    pck_pth = 'E:/dynamic_xray/data/JapeneseData_processed/tsne.p'
    flds = ['E:/dynamic_xray/data/JapeneseData_processed/numpy','E:/dynamic_xray/data/JapeneseData_processed/numpy']
    fls = [xx for fld in flds for xx in listall(fld)]

    #build DF
    df = pd.DataFrame(columns = [])


    #####tmp
    #####tmp
    #####tmp
    #fls = fls[:10]


    #load
    arrs = [np.load(fl) for fl in fls]

    #colors based on lenght of array
    discrete_colors= len(arrs)
    c = [xx.shape[1] for xx in arrs]
    clst = [[plt.cm.Spectral(np.linspace(0,1,discrete_colors)[i])]*c[i] for i in range(discrete_colors)]
    clst = [yy for xx in clst for yy in xx]


    #tnse
    svpth = 'E:/dynamic_xray/data/kde.jpeg'
    X = np.nan_to_num(np.concatenate(arrs, axis=1).swapaxes(0,1),nan=0)
    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=40, learning_rate=500, n_iter=1000)
    Y=tsne.fit_transform(X)
    plt.scatter(Y[:,0], Y[:,1], c=clst, alpha=0.1,s=2)

    if False:
        for early_exaggeration in [12, 48]:
            for perplexity in [75]:
                for learning_rate in [100, 1000]:
                    Y = TSNE(n_components=2, early_exaggeration=early_exaggeration, perplexity=perplexity, learning_rate=learning_rate).fit_transform(X)
                    plt.figure();plt.scatter(Y[:,0], Y[:,1], c=clst, alpha=0.5,s=2)
                    plt.title('early_exaggeration {}, perplexity {}, learning_rate {}'.format(early_exaggeration, perplexity, learning_rate))

    g=sns.kdeplot(x=Y[:,0], y=Y[:,1], fill=True, thresh=.01, levels=1000, cmap='viridis') #cmap="viridis")
    g.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    g.tick_params(bottom=False, left=False)
    sns.despine(top=True, bottom=True, left=True, right=True)
    plt.savefig(svpth, dpi=450)

    dct = {'tsne':tsne, 'X':X, 'Y':Y, 'svpth': svpth}

    with open(pck_pth, 'wb') as handle:
        pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    import copy
    my_cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    my_cmap.set_under('w')
    x,y=Y[:,0], Y[:,1]
    svpth = 'E:/dynamic_xray/sleap_output/kde.jpeg'
    plt.hist2d(x, y, bins=(60, 60),vmin=.01, cmap=my_cmap)
    g.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    g.tick_params(bottom=False, left=False)
    sns.despine(top=True, bottom=True, left=True, right=True)
    plt.savefig(svpth, dpi=450)

    '''
    from scipy.stats import kde
    import copy
    my_cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    my_cmap.set_under('w')
    nbins=300
    x,y=Y[:,0], Y[:,1]
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # Make the plot
    zz=zi.reshape(xi.shape)
    g=plt.pcolormesh(xi, yi, zz, shading='auto', rasterized=True, vmin=zz.min()+.0000001, vmax=zz.max(), cmap=my_cmap)#, vmin=100)
    plt.axis('off')
    sns.despine(top=True, bottom=True, left=True, right=True)
    plt.savefig(svpth, dpi=450)

    # Change color palette
    #plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
    #plt.show()


    '''
    #watershed
    from scipy import ndimage
    from skimage import morphology
    from skimage.feature import peak_local_max
    from skimage import color, io
    im = color.rgb2gray(io.imread(svpth))
    #im = plt.imread(svpth)[:,:,0]
    distance = ndimage.distance_transform_edt(im)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=im)
    markers = morphology.label(local_maxi)
    labels_ws = morphology.watershed(-distance, markers, mask=im)
    plt.matshow(labels_ws)
    '''
    #https://github.com/abidrahmank/OpenCV2-Python-Tutorials/blob/master/source/py_tutorials/py_imgproc/py_watershed/py_watershed.rst
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    img = cv2.imread(svpth)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    plt.imshow(img)
    plt.imshow(markers)

    #%%tsne videos
    #load
    concatdf = 'E:/dynamic_xray/data/concatenated_dataframe.csv'
    pck_pth = 'E:/dynamic_xray/data/JapeneseData_processed/tsne.p'
    with open(pck_pth, 'rb') as handle:
        dct = pickle.load(handle)

    X=dct['X']; Y=dct['Y']; svpth=dct['svpth'], tsne=dct['tsne']
    df = pd.read_csv(concatdf)
    #make tnse video
    kdeplot = dc

    #%%
    #%%
    #%%
    #%%
