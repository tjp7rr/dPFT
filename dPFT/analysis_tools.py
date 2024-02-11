import matplotlib
import os, pandas as pd, numpy as np, pickle as pckl
from scipy.stats import ttest_ind, pearsonr
import matplotlib.pyplot as plt, seaborn as sns, json, matplotlib as mpl,imageio
from sklearn import metrics
import sklearn, scipy
from scipy.signal import argrelextrema
import itertools
import shutil

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
pd.pandas.set_option('display.max_columns', None)
savefig = True
pd.set_option("display.max_rows", None, "display.max_columns", None)

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

def point_distance(xy0, xy1):
    return np.sqrt( (xy0[0] - xy1[0])**2 + (xy0[1] - xy1[1])**2 )

def find_peaks_relative_to_other_peaks(x,y, number_points_before_after=4, alpha=1, gaussigma=4, figdst=False, savedst = False, pntid = 'peaks', add_text_values=True, filetype='.png', cat_class=False, cm_to_pixel_conversion=False, area_type='lungs-heart'):
    '''

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    number_points_before_after : TYPE, optional
        DESCRIPTION. The default is 4.
        # number of points to be checked before and after

    Returns
    -------
    None.

    '''
    if False:
        # Generate a noisy AR(1) sample
        np.random.seed(0)
        rs = np.random.randn(200)
        y = [0]
        for r in rs:
            y.append(y[-1] * 0.9 + r)
        x = list(range(0,201))

    #if staying with pixels convert to 10^3 pixels
    if not cm_to_pixel_conversion: y = y/1000

    #filter
    ygaus = scipy.ndimage.gaussian_filter1d(y, sigma=gaussigma)

    #build df
    df = pd.DataFrame(data=np.asarray([y,ygaus,x]).T, columns=['rawdata', 'gausdata', 'frame'])

    #convert from pixels to distance
    if cm_to_pixel_conversion:
        df['rawdata'] = df['rawdata'] * cm_to_pixel_conversion
        df['gausdata'] = df['gausdata'] * cm_to_pixel_conversion
        distance_label = 'cm^2'
    else:
        distance_label = 'Pixels (1000)'

    df['units'] = distance_label
    df['area_type'] = area_type

    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df['gausdata'].values, np.less_equal, order=number_points_before_after)[0]]['gausdata']
    df['max'] = df.iloc[argrelextrema(df['gausdata'].values, np.greater_equal, order=number_points_before_after)[0]]['gausdata']

    #keep peaks/valleys, find max-min deltas
    mx = df[df.columns[:3].tolist()+['max']].dropna(axis=0); mx['extrema'] = 'max'
    mn = df[df.columns[:3].tolist()+['min']].dropna(axis=0); mn['extrema'] = 'min'
    ndf = pd.concat([mx,mn]); del ndf['max']; del ndf['min']
    ndf = ndf.sort_values('frame')
    ndf['extrema_gaus_deltas'] = ndf['gausdata'].diff()
    ndf['extrema_raw_deltas'] = ndf['rawdata'].diff()
    ndf['min_lung_area'] = np.round(df['gausdata'].min(),4)
    ndf['max_lung_area'] = np.round(df['gausdata'].max(),4)
    ndf['file'] = pntid

    #include mx and mn info
    peaks_med = mx['gausdata'].median()
    peaks_min = mx['gausdata'].min()
    peaks_ave = mx['gausdata'].mean()
    troughs_med = mn['gausdata'].median()
    troughs_max = mn['gausdata'].max()
    troughs_ave = mn['gausdata'].mean()

    #take median tv and max vc
    tv_median = ndf[ndf['extrema_gaus_deltas']>0]['extrema_gaus_deltas'].median()
    vc_max = ndf[ndf['extrema_gaus_deltas']>0]['extrema_gaus_deltas'].max()
    vc = df['max'].max() - df['min'].min()
    tlc = mx['gausdata'].max()
    pulmreserve = 100*(1-(tv_median/vc))

    roundto=1
    text = 'TLC {}\n\nTV (median TD) {}\n\nVC {}\n\n, Pulmonary Reserve %\n {}%\n\nMax TD {}\n\nTidal diffs:\n {}'.format(
        np.round(tlc , roundto),
        np.round(tv_median,roundto),
        np.round(vc, roundto),
        np.round(pulmreserve, roundto),
        np.round(vc_max, roundto),
        list(np.round(ndf[ndf['extrema_gaus_deltas']>0]['extrema_gaus_deltas'].values,roundto)),)
    #look at tidal volumes
    #sns.histplot(ndf[ndf['extrema_deltas']>0]['extrema_deltas'], bins=20)

    # Plot results
    plt.scatter(df.index, df['min'], c='blue', alpha=alpha, label='min')
    plt.scatter(df.index, df['max'], c='green', alpha=alpha, label='max')
    plt.plot(df.index, df['gausdata'], c='black', alpha=alpha, label='Gaussian')
    plt.plot(df.index, df['rawdata'], c='red', alpha=alpha, label = 'raw')

    plt.plot([-10]*len(df['max']), df['max'], '_', color='green')
    plt.plot([-10]*len(df['min']), df['min'], '_', color='blue')

    plt.xlabel('Frame'); plt.ylabel(distance_label)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if add_text_values: plt.text(.75, 0.15, text, fontsize=8, transform=plt.gcf().transFigure, wrap=True)
    if cat_class:
        plt.title('{}\n{}\nAreas calculated using {}'.format(pntid, cat_class, area_type))
    else:
        plt.title('{}\nAreas calculated using {}'.format(pntid, area_type))
    plt.tight_layout()
    sns.despine(top=True, right=True, left=False, bottom=False)

    if figdst: plt.savefig(os.path.join(figdst, pntid+filetype), dpi=300, transparent = True); plt.close()
    if savedst: ndf.to_csv(os.path.join(savedst, pntid+'_peaks_dataframe.csv'))


    #make flow curves using ygaus: y=flow, x=volume
    import matplotlib as mpl
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    df['flow'] = -df['gausdata'].diff() #need this to be negative as positive flow is considered expiratory
    df['flowgaus'] = scipy.ndimage.gaussian_filter1d(df['flow'], sigma=gaussigma)


    #build out a flow dataframe using flowgaus, not flow. This underestimates effects like peak flow but protects against CNN having errors in single frames
    varr = 'flowgaus' #'flow' #flowgaus
    flowmax = df[varr].max()
    flowmin = df[varr].min()
    flowmed = df[varr].median()
    flowave = df[varr].mean()
    flowmedabs = df[varr].abs().median()
    flowaveabs = df[varr].abs().mean()

    fdf = pd.DataFrame(data = [[flowmax, flowmin, flowmed, flowave, flowmedabs, flowaveabs, peaks_med, peaks_min, peaks_ave, troughs_med, troughs_max, troughs_ave, tv_median, vc, tlc, pulmreserve]], columns = ['flowmax', 'flowmin', 'flowmed', 'flowave', 'flowmedabs', 'flowaveabs', 'peaks_med', 'peaks_min', 'peaks_ave', 'troughs_med', 'troughs_max', 'troughs_ave', 'tv_median', 'VC', 'TLC', 'PulmReserve'])
    fdf['extrema_flow_deltas'] = flowmax - flowmin
    fdf['file'] = pntid

    if savedst:
        fdf.to_csv(os.path.join(savedst, pntid+'_flow_dataframe.csv'))
        df.to_csv(os.path.join(savedst, pntid+'_original_data_w_gaussian_dataframe.csv'))

    #flow plots
    yy,xx,tt = df[['flow', 'gausdata','frame']].values[1:].T
    my_cmap = plt.cm.cool(np.arange(len(yy)))
    #1
    dydx = np.asarray(list(range(0, len(xx))))#np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
    points = np.array([tt, xx]).T.reshape(-1, 1, 2)
    segments0 = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, axs = plt.subplots(2, 1)
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments0, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    fig.colorbar(line, ax=axs[0]).set_label('Frame')
    if cat_class:
        axs[0].set_title(pntid+'    '+cat_class)
    else:
        axs[0].set_title(pntid)

    axs[0].set_ylabel(distance_label)
    axs[0].set_xlabel('Frame')

    #2
    dydx = np.asarray(list(range(0, len(xx))))#np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
    pointss = np.array([xx, yy]).T.reshape(-1, 1, 2)
    segments1 = np.concatenate([pointss[:-1], pointss[1:]], axis=1)
    #fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    # Create a continuous norm to map from data points to colors
    lcc = LineCollection(segments1, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lcc.set_array(dydx)
    lcc.set_linewidth(2)
    lines = axs[1].add_collection(lcc)
    fig.colorbar(lines, ax=axs[1]).set_label('Frame')
    axs[1].set_ylabel('Flow')
    axs[1].set_xlabel(distance_label)


    axs[0].set_xlim(tt.min(), tt.max())
    axs[0].set_ylim(xx.min()*0.97, xx.max()*1.03)
    axs[1].set_xlim(xx.min()*0.99, xx.max()*1.01)
    axs[1].set_ylim(yy.min()*1.1, yy.max()*1.1)
    fig.tight_layout(pad=1.0)
    sns.despine(top=True, right=True, left=False, bottom=False)
    if figdst: plt.savefig(os.path.join(figdst, pntid+'_flow_plots'+filetype), dpi=300, transparent = True); plt.close()
    return


def plot_triggered_breaths(peaksdf, areadf, gausdf, figdst=False, pntid = 'triggered', filetype='.png'):
    '''
    peaksdf="C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data/peaks_202202/PatientID_0001_Age_054Y_DDRDate_20200224_PA_peaks_dataframe.csv"
    areadf = 'E:/dynamic_xray/processed/202202_data/PatientID_0001_Age_054Y_DDRDate_20200224_PA/0001_PA_clahe_areas_and_curvature_indices.csv'
    gausdf="C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data/peaks_202202/PatientID_0001_Age_054Y_DDRDate_20200224_PA_original_data_w_gaussian_dataframe.csv"
    '''
    pdf = pd.read_csv(peaksdf)
    gdf = pd.read_csv(gausdf); gdf = gdf[gdf.columns[1:]]; gdf = gdf.sort_values('frame')
    adf = pd.read_csv(areadf); adf = adf.sort_values('frame')
    adf['gausdata'] = gdf['gausdata']; adf['flow'] = gdf['flow']; adf['flowgaus'] = gdf['flowgaus']
    units = pd.read_csv(gausdf)['units'][0]

    #parse into min->max bins
    mx_frames = pdf[pdf.extrema=='max']['frame'].values.astype('int')
    mn_frames = pdf[pdf.extrema=='min']['frame'].values.astype('int')

    #now make into time bins
    frames = adf.frame.unique(); frames.sort()
    min_trig_frames = np.zeros(frames.shape[0]).astype('int')
    min_idx = np.zeros(frames.shape[0]).astype('object')
    iterr = 1
    #first each min to max (inhale)
    for i,mn in enumerate(mn_frames):
        #find next max
        mx = mx_frames[np.argmax(mx_frames>mn)]
        min_trig_frames[mn:mx] = range(1, 1+mx-mn)
        min_idx[mn:mx] = 'Inhale {}'.format(iterr)
        iterr+=1

    #now for each mx next min (exhale)
    max_trig_frames = np.zeros(frames.shape[0]).astype('int')
    max_idx = np.zeros(frames.shape[0]).astype('object')
    iterr = 1
    for i,mx in enumerate(mx_frames):
        #find next max
        mn = mn_frames[np.argmax(mn_frames>mx)]
        max_trig_frames[mx:mn] = range(1, 1+mn-mx)
        max_idx[mx:mn] = 'Exhale {}'.format(iterr)
        iterr+=1

    #now add to adf
    adf['min_trig_frames'] = min_trig_frames
    adf['min_idx'] = min_idx
    adf['max_trig_frames'] = max_trig_frames
    adf['max_idx'] = max_idx

    p0 = sns.color_palette("Blues_r", len(adf['min_idx'].unique())*2)[:len(adf['min_idx'].unique())-1]
    p1 = sns.color_palette("Reds_r", len(adf['max_idx'].unique())*2)[:len(adf['max_idx'].unique())-1]

    #now plot
    fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(12,6))
    sns.lineplot(data=adf[adf['min_idx']!=0], x="min_trig_frames", y="gausdata", hue="min_idx", ax=ax[0], palette=p0, alpha=0.7)
    sns.lineplot(data=adf[adf['max_idx']!=0], x="max_trig_frames", y="gausdata", hue="max_idx", ax=ax[0], palette=p1, alpha=0.7)
    ax[0].set_title(pntid)
    ax[0].legend(title='Breath', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax[0].set_xlabel('Triggered Frame')
    ax[0].set_ylabel(units)

    #selecting flow data not gaus flow data because flow data is determined via gaussian areas*****
    sns.lineplot(data=adf[adf['min_idx']!=0], x="min_trig_frames", y="flow", hue="min_idx", ax=ax[1], palette=p0, alpha=0.7)
    sns.lineplot(data=adf[adf['max_idx']!=0], x="max_trig_frames", y="flow", hue="max_idx", ax=ax[1], palette=p1, alpha=0.7)
    ax[1].legend(title='Breath', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax[1].set_xlabel('Triggered Frame')
    ax[1].set_ylabel('Flow ({}/dt)'.format(units))
    fig.tight_layout(pad=1.0)
    sns.despine(top=True, right=True, left=False, bottom=False)
    if figdst: plt.savefig(os.path.join(figdst, pntid+'_triggered_plots'+filetype), dpi=300, transparent = True); plt.close()
    return

#%%
def generated_triggered_breaths_df(peaksdf, areadf, gausdf, figdst=False, pntid = 'triggered', filetype='.png'):
    '''
    peaksdf="C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data/peaks_202202/PatientID_0001_Age_054Y_DDRDate_20200224_PA_peaks_dataframe.csv"
    areadf = 'E:/dynamic_xray/processed/202202_data/PatientID_0001_Age_054Y_DDRDate_20200224_PA/0001_PA_clahe_areas_and_curvature_indices.csv'
    gausdf="C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data/peaks_202202/PatientID_0001_Age_054Y_DDRDate_20200224_PA_original_data_w_gaussian_dataframe.csv"

    just run code below once

    '''
    prefix = 'C:/Users/tpisano/Google Drive/'
    figdst = os.path.join(prefix, 'Documents/Python_Scripts/dxr_project/figures/202205/'); makedir(figdst)
    savedst = os.path.join(prefix, 'Documents/Python_Scripts/dxr_project/data/peaks_202205'); makedir(savedst)
    src = 'E:/dynamic_xray/processed/202204_data'
    #src = os.path.join(prefix, 'Documents/Python_Scripts/dxr_project/data/202205_data/')
    gaussigma = 3
    alpha = 0.5
    number_points_before_after=15
    fls = listall(src, keyword='_clahe_areas_and_curvature_indices.csv')
    filetype='.pdf' #'.png'
    cat_class = True


    #triggered
    aflsrc = 'E:/dynamic_xray/processed/202205_data'
    afls = listall(src, keyword='_clahe_areas_and_curvature_indices.csv')
    gpflsrc = "C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data/peaks_202205/"
    gfls = listall(gpflsrc, keyword='original_data_w_gaussian_dataframe.csv')
    pfls = listall(gpflsrc, keyword='peaks_dataframe.csv')

    bigdata=[]

    for i, afl in enumerate(afls):
        print(i)
        pntid = os.path.basename(os.path.dirname(afl))
        gfl = [xx for xx in gfls if pntid in xx][0]
        pfl = [xx for xx in pfls if pntid in xx][0]
        #if 'PatientID_353990_Age_048Y_DDRDate_20200218_PA' == pntid: break
        peaksdf=pfl; areadf=afl; gausdf=gfl


        pdf = pd.read_csv(peaksdf)
        gdf = pd.read_csv(gausdf); gdf = gdf[gdf.columns[1:]]; gdf = gdf.sort_values('frame')
        adf = pd.read_csv(areadf); adf = adf.sort_values('frame')
        adf['gausdata'] = gdf['gausdata']; adf['flow'] = gdf['flow']; adf['flowgaus'] = gdf['flowgaus']

        #parse into min->max bins
        mx_frames = pdf[pdf.extrema=='max']['frame'].values.astype('int')
        mn_frames = pdf[pdf.extrema=='min']['frame'].values.astype('int')

        #now make into time bins
        frames = adf.frame.unique(); frames.sort()
        min_trig_frames = np.zeros(frames.shape[0]).astype('int')
        min_idx = np.zeros(frames.shape[0]).astype('object')
        iterr = 1
        #first each min to max (inhale)
        for i,mn in enumerate(mn_frames):
            #find next max
            mx = mx_frames[np.argmax(mx_frames>mn)]
            min_trig_frames[mn:mx] = range(1, 1+mx-mn)
            min_idx[mn:mx] = 'Inhale {}'.format(iterr)
            iterr+=1

        #now for each mx next min (exhale)
        max_trig_frames = np.zeros(frames.shape[0]).astype('int')
        max_idx = np.zeros(frames.shape[0]).astype('object')
        iterr = 1
        for i,mx in enumerate(mx_frames):
            #find next max
            mn = mn_frames[np.argmax(mn_frames>mx)]
            max_trig_frames[mx:mn] = range(1, 1+mn-mx)
            max_idx[mx:mn] = 'Exhale {}'.format(iterr)
            iterr+=1

        #now add to adf
        adf['min_trig_frames'] = min_trig_frames
        adf['min_idx'] = min_idx
        adf['max_trig_frames'] = max_trig_frames
        adf['max_idx'] = max_idx
        adf['pntid'] = pntid

        #for each inhale and exhale need to rank by extreme (area and flow) for each pntid
        cats = ['rank_inhale_flow', 'rank_exhale_flow', 'rank_inhale_volume', 'rank_exhale_volume']

        #generate ranks for inhale/exhale flow/area (remember flow is calculated using gausdata, so don't use flow gaus)
        #exhales
        exhale_area_maxes = pd.DataFrame(data=[[exhale, adf[adf['max_idx']==exhale]['gausdata'].max()] for exhale in adf['max_idx'].unique()], columns=['hale #', 'value'])
        exhale_area_maxes = exhale_area_maxes[exhale_area_maxes['hale #']!=0.0].sort_values('value', ascending=False)
        exhale_area_maxes['rank'] = range(1, 1+len(exhale_area_maxes))
        exhale_area_maxes=exhale_area_maxes.append(pd.DataFrame(data = [[0.0, 0.0, -1]], columns=exhale_area_maxes.columns), ignore_index=True)

        exhale_flow_maxes = pd.DataFrame([[exhale, adf[adf['max_idx']==exhale]['flow'].max()] for exhale in adf['max_idx'].unique()], columns=['hale #', 'value'])
        exhale_flow_maxes = exhale_flow_maxes[exhale_flow_maxes['hale #']!=0.0].sort_values('value', ascending=False)
        exhale_flow_maxes['rank'] = range(1, 1+len(exhale_flow_maxes))
        exhale_flow_maxes=exhale_flow_maxes.append(pd.DataFrame(data = [[0.0, 0.0, -1]], columns=exhale_flow_maxes.columns), ignore_index=True)

        #inhales
        inhale_area_maxes = pd.DataFrame([[inhale, adf[adf['min_idx']==inhale]['gausdata'].max()] for inhale in adf['min_idx'].unique()], columns=['hale #', 'value'])
        inhale_area_maxes = inhale_area_maxes[inhale_area_maxes['hale #']!=0.0].sort_values('value', ascending=False)
        inhale_area_maxes['rank'] = range(1, 1+len(inhale_area_maxes))
        inhale_area_maxes=inhale_area_maxes.append(pd.DataFrame(data = [[0.0, 0.0, -1]], columns=inhale_area_maxes.columns), ignore_index=True)

        inhale_flow_maxes = pd.DataFrame([[inhale, adf[adf['min_idx']==inhale]['flow'].max()] for inhale in adf['min_idx'].unique()], columns=['hale #', 'value'])
        inhale_flow_maxes = inhale_flow_maxes[inhale_flow_maxes['hale #']!=0.0].sort_values('value', ascending=False)
        inhale_flow_maxes['rank'] = range(1, 1+len(inhale_flow_maxes))
        inhale_flow_maxes=inhale_flow_maxes.append(pd.DataFrame(data = [[0.0, 0.0, -1]], columns=inhale_flow_maxes.columns), ignore_index=True)

        #apply ranks
        adf['rank_exhale_area'] = adf.apply(lambda xx:exhale_area_maxes[exhale_area_maxes['hale #']==xx['max_idx']]['rank'].values[0] ,1)
        adf['rank_exhale_flow'] = adf.apply(lambda xx:exhale_flow_maxes[exhale_flow_maxes['hale #']==xx['max_idx']]['rank'].values[0] ,1)
        adf['rank_inhale_area'] = adf.apply(lambda xx:inhale_area_maxes[inhale_area_maxes['hale #']==xx['min_idx']]['rank'].values[0] ,1)
        adf['rank_inhale_flow'] = adf.apply(lambda xx:inhale_flow_maxes[inhale_flow_maxes['hale #']==xx['min_idx']]['rank'].values[0] ,1)

        bigdata.append(adf)
    bigdf = pd.concat(bigdata)
    bigdf.to_csv('C:/Users/tpisano/Google Drive/Documents/Python_Scripts/dxr_project/data/combined_flow_dataframe_with_triggers.csv')

    #add categories to bigdf
    #merged = pd.read_csv(os.path.join(prefix, 'Documents/Python_Scripts/dxr_project/data/merged_cnn_and_pft_data_reformatted_202205.csv'))
    bigdf = pd.read_csv(os.path.join(prefix, 'Documents/Python_Scripts/dxr_project/data/combined_flow_dataframe_with_triggers.csv'))
    merged = pd.read_csv(os.path.join(prefix, 'Documents/Python_Scripts/dxr_project/data/Updated_file_w_new_PFT_data_2023.csv'))
    mapper = {pntid: [pntid.split('_')[1], pntid.split('_')[3]] for pntid in bigdf.pntid.unique()}
    pntid_class_map = {}
    for k,v in mapper.items():
        try:
            pntid_class_map[k] = str(merged[(merged['Patient ID']==v[0])&(merged['Age']==v[1])]['PulmClass_2023'].values[0])
        except:
            pntid_class_map[k] = 'nan'

    #these are all true, that could change
    pntid_screen_dct = {'PatientID_{}_Age_{}_DDRDate_{}_PA'.format(row['Patient ID'],row['Age'],row['DDR Date_x']):row['screened_in'] for i,row in merged.iterrows()}

    bigdf['screened_in'] = bigdf.apply(lambda xx: True if xx['pntid'] in pntid_screen_dct.keys() else False, 1)

    #pulm class
    bigdf['PulmClass'] = bigdf.apply(lambda xx: pntid_class_map[xx['pntid']], 1)

    #replace unclassfied pulm class that were nans
    bigdf['PulmClass'] = bigdf['PulmClass'].replace({'nan': 'Unclassified'})

    bigdf.to_csv(os.path.join(prefix, 'Documents/Python_Scripts/dxr_project/data/combined_flow_dataframe_with_triggers.csv'))
#%%

if __name__ == "__main__":
    
    # set paths
    dxr_fld = r'C:\Users\tpisano\Downloads\dPFT'#'/Users/tjp7rr1/Downloads/dPFT' #'E:/dPFT' 
    fl = os.path.join(dxr_fld, 'output/tracking_output_analysis/1_PA_clahe_areas_and_curvature_indices.csv') #listall(dxr_fld, keyword='_clahe_areas_and_curvature_indices.csv')[0]
    
    # switches
    gaussigma = 3
    alpha = 0.5
    number_points_before_after=15
    filetype= '.png' #'.pdf' produces better images but not as easy to read into jupyter notebook
    cat_class = False 
    area_type = 'total_lung' # between 'lungs-heart' and 'total_lung'
    cm_to_pixel_conversion=.0016 #cm_to_pixel_conversion - this would be square cm^2 per pixel. E.g. 400um x 400um pixel = 0.0016
    
    # folder set up
    figdst = os.path.join(dxr_fld, 'output/figures'); makedir(figdst)
    savedst = os.path.join(dxr_fld, 'output/peaks'); makedir(savedst)
        
    # find peaks relative to other peaks
    df = pd.read_csv(fl).sort_values('frame')
    x,y=df[['frame', area_type]].values.T 
    pntid = '1_PA'
    find_peaks_relative_to_other_peaks(x,y, number_points_before_after=number_points_before_after, gaussigma=gaussigma, alpha=alpha, figdst=figdst, savedst=savedst, pntid=pntid, filetype=filetype, cat_class=cat_class, cm_to_pixel_conversion=cm_to_pixel_conversion, area_type=area_type)
    
    # now lets take a look at the plots (analysis csv files will be saved under peaks)
    # first our dPFT lung area over time plot
    plt.imshow(imageio.imread(os.path.join(figdst, '1_PA.png'))); plt.axis('off')
    
    # we can also take a look at the flow plots as well 
    plt.imshow(imageio.imread(os.path.join(figdst, '1_PA_flow_plots.png'))); plt.axis('off')
    
    #Now we can also generate breath triggered plots
    gfl = os.path.join(savedst, '1_PA_original_data_w_gaussian_dataframe.csv')
    pfl = os.path.join(savedst, '1_PA_peaks_dataframe.csv')
    plot_triggered_breaths(peaksdf=pfl, areadf=fl, gausdf=gfl, figdst=figdst, pntid=pntid, filetype=filetype)

    ####
    # now lets take a look at our breath triggered plots
    plt.imshow(imageio.imread(os.path.join(figdst, '1_PA_triggered_plots.png'))); plt.axis('off')
