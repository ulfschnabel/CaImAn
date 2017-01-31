# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:55:48 2016

@author: agiovann
"""

#%%
from skimage.external import tifffile
import caiman as cm
from sklearn.decomposition import PCA,DictionaryLearning,MiniBatchDictionaryLearning,NMF
from sklearn.decomposition import FastICA
import pylab as pl
import glob
import numpy as np
#import spams
#from PIL import Image
import time
#%%
a = cm.load('demoMovie.tif')[:,20:50,20:50]
Yr = cm.movie.to_2D(a)
mov = Yr
#%%
res = compute_event_exceptionality(Yr.T)
mns = -cm.movie(np.reshape(res[1],[30,30,-1]).transpose([2,1,0]))
mns[mns > 1000] = 1000
mov = cm.movie.to_2D(mns)

#%%
noise = res[2]
mode = cm.components_evaluation.mode_robust(Yr,0)
Yr_1 = (cm.movie.to_2D(a)-mode)/res[2]
mns_1 = (np.reshape(Yr_1,[-1,30,30],order='F'))
mov = np.maximum(0,cm.movie.to_2D(mns_1))
#%%


n_comps = 10
pca = PCA(n_comps)
pca = NMF(n_comps,alpha = 10,l1_ratio = 1,init = 'nndsvda')

pca.fit(mov)
#%%
import cv2
comps = np.reshape(pca.components_,[n_comps,30,30])
for count,comp in enumerate(comps):
    pl.subplot(4,4,count+1)
    blur = cv2.GaussianBlur(comp.astype(np.float32),(5,5),0)
    blur = np.array(blur/np.max(blur) * 255, dtype = np.uint8)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    pl.imshow((th3*comp).T)

#%%
n_comps = 3
dl = DictionaryLearning(n_comps,alpha=1,verbose = True)
comps = dl.fit_transform(Yr.T)
comps = np.reshape(comps,[30,30,n_comps]).transpose([2,0,1])
for count,comp in enumerate(comps):
    pl.subplot(4,4,count+1)
    pl.imshow(comp)
#%%
N_ICA_COMPS = 8
ica =  FastICA(N_ICA_COMPS,max_iter=10000,tol= 10e-8)
ica.fit(pca.components_)
#%
comps = np.reshape(ica.components_,[N_ICA_COMPS,30,30])
for count,comp in enumerate(comps):
    idx = np.argmax(np.abs(comp))
    comp = comp * np.sign(comp.flatten()[idx])
    pl.subplot(4,4,count+1)
    pl.imshow(comp.T)
#%%
n_comps = 5    
a = cm.load('demoMovie.tif')[:100,20:50,20:50].IPCA_stICA(15,n_comps) 
#%%
comps = np.reshape(a,[n_comps,30,30])
for count,comp in enumerate(comps):
    pl.subplot(3,4,count+1)
    pl.imshow(comp)
#%%
m = cm.load('demoMovie.tif')
m1 = m.IPCA_denoise(50)  
#%%
m1.play(fr = 30,gain = 5.,magnification=4) 
#%%
m1.save('demoMovieDen.tif')
#%%
m1 = cm.load('demoMovieDen.tif')

m1.play(fr = 30,gain = 4.,magnification=4) 


#%%

fls = glob.glob('*Den.tif')
print fls
all_names, all_shifts, all_xcorrs, all_templates = cm.motion_correction.motion_correct_online_multifile(fls,0)

#%%
Yr, dims, T = cm.load_memmap(all_names[0])
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%%
K = 35  # number of neurons expected per patch
gSig = [7, 7]  # expected half size of neurons
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 2  # order of the autoregressive system
cnm = cnmf.CNMF(8,  k=K, gSig=gSig, merge_thresh=merge_thresh,
                p=p, dview=None, Ain=None,method_deconvolution='oasis')
cnm = cnm.fit(images)
#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%%
final_frate = 10
tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, remove_baseline=True,
                                      N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

idx_components_r = np.where(r_values >= .5)[0]
idx_components_raw = np.where(fitness_raw < -40)[0]
idx_components_delta = np.where(fitness_delta < -20)[0]




idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_components_bad = np.setdiff1d(range(len(traces)), idx_components)

print(' ***** ')
print len(traces)
print(len(idx_components))


#%% visualize components
# pl.figure();
Cn = cm.local_correlations(Y[:,:,:3000])
crd = plot_contours(A.tocsc(), np.mean(images,0), thr=0.9)
    
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
                               idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)