# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

"""
#%%
try:
    %load_ext autoreload
    %autoreload 2
    print 1
except:
    print 'NOT IPYTHON'

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
#plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

#sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import glob
import os
import scipy
from ipyparallel import Client
#%%
#backend='SLURM'
backend='local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
#%% start cluster for efficient computation
single_thread=False

if single_thread:
    dview=None
else:    
    try:
        c.close()
    except:
        print 'C was not existing, creating one'
    print "Stopping  cluster to avoid unnencessary use of memory...."
    sys.stdout.flush()  
    if backend == 'SLURM':
        try:
            cse.utilities.stop_server(is_slurm=True)
        except:
            print 'Nothing to stop'
        slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cse.utilities.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)        
    else:
        cse.utilities.stop_server()
        cse.utilities.start_server()        
        c=Client()

    print 'Using '+ str(len(c)) + ' processes'
    dview=c[:len(c)]
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames=[]
base_folder='./movies' # folder containing the demo files
for file in glob.glob(os.path.join(base_folder,'*.tif')):
    if file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
fnames=fnames
#%% Create a unique file fot the whole dataset
# THIS IS  ONLY IF YOU NEED TO SELECT A SUBSET OF THE FIELD OF VIEW 
#fraction_downsample=1;
#idx_x=slice(10,502,None)
#idx_y=slice(10,502,None)
#fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),remove_init=0,idx_xy=(idx_x,idx_y))

#%%
#idx_x=slice(12,500,None)
#idx_y=slice(12,500,None)
#idx_xy=(idx_x,idx_y)
downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
idx_xy=None
base_name='Yr'
name_new=cse.utilities.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy )
name_new.sort(key=lambda fn: np.int(fn[fn.find(base_name)+len(base_name):fn.find('_')]))
print name_new
#%%
n_chunks=6 # increase this number if you have memory issues at this point
ls=cse.utilities.save_memmap_join(name_new,base_name='Yr', n_chunks=6, dview=dview)
#%%  Create a unique file fot the whole dataset
##
#fraction_downsample=1; # useful to downsample the movie across time. fraction_downsample=.1 measn downsampling by a factor of 10
#fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),order='F')
#%%

#%%
#fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr,dims,T=cse.utilities.load_memmap(fname_new)
d1,d2=dims
images=np.reshape(Yr.T,[T]+list(dims),order='F')
Y=np.reshape(Yr,dims+(T,),order='F')
#%%
Cn = cse.utilities.local_correlations(Y[:,:,:3000])
pl.imshow(Cn,cmap='gray')  
#%%
rf=15 # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 4 #amounpl.it of overlap between the patches in pixels    
K=5 # number of neurons expected per patch
gSig=[7,7] # expected half size of neurons
merge_thresh=0.8 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
memory_fact=1; #unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results=False
#%% RUN ALGORITHM ON PATCHES
cnmf=cse.CNMF(n_processes, k=K,gSig=gSig,merge_thresh=0.8,p=0,dview=None,Ain=None,rf=rf,stride=stride, memory_fact=memory_fact,\
        method_init='sparse_nmf',alpha_snmf=1e1,only_init_patch=True)
cnmf=cnmf.fit(images)

A_tot=cnmf.A
C_tot=cnmf.C
YrA_tot=cnmf.YrA
b_tot=cnmf.b
f_tot=cnmf.f
sn_tot=cnmf.sn

print 'Number of components:' + str(A_tot.shape[-1])
#%%
A=A_tot
C=C_tot
merged_ROIs=[0]
while len(merged_ROIs)>0:
    A,C,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr, A, b_tot, C, f_tot, None, sn_tot, options['temporal_params'], options['spatial_params'], dview=c[:], thr=merge_thresh, fast_merge=True)

#%%
idx_components, fitness, erfc ,r_values, num_significant_samples = cse.utilities.evaluate_components(Y,C_tot+YrA_tot,A_tot,N=5,robust_std=True,thresh_finess=-10,compute_r_values=False)
sure_in_idx= idx_components[fitness<-10]

print ('Keeping ' + str(len(sure_in_idx)) + ' components out of ' + str(len(idx_components)))
#%%
pl.figure()
crd = cse.utilities.plot_contours(A_tot.tocsc()[:,sure_in_idx],Cn,thr=0.9)
#%%
A_tot=A_tot.tocsc()[:,sure_in_idx]
C_tot=C_tot[sure_in_idx]
#%%
save_results = True
if save_results:
    np.savez('results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2,b=b,f=f) 
#%% if you have many components this might take long!
pl.figure()
crd = cse.utilities.plot_contours(A_tot,Cn,thr=0.9)
#%%
cnmf=cse.CNMF(n_processes, k=A_tot.shape,gSig=gSig,merge_thresh=merge_thresh,p=p,dview=dview,Ain=A_tot,Cin=C_tot,\
                 f_in=f_tot, rf=None,stride=None)
cnmf=cnmf.fit(images)
#%%
A=cnmf.A
C=cnmf.C
YrA=cnmf.YrA
b=cnmf.b
f=cnmf.f
sn=cnmf.sn
#%%
options = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=A.shape[-1],thr=merge_thresh)

#%%

merged_ROIs=[0]
while len(merged_ROIs)>0:
    A,C,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr, A, b, C, f, None, sn, options['temporal_params'], options['spatial_params'], dview=c[:], \
    thr=0.7, fast_merge=True)
#%%
options['temporal_params']['p']=0
options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
options['temporal_params']['backend']='ipyparallel'
C,f,S,bl,c1,neurons_sn,g2,YrA = cse.temporal.update_temporal_components(Yr,A,np.atleast_2d(b),C,f,dview=c[:],**options['temporal_params'])
    
#%%
    
#%%
traces=C+YrA
num_sampls=traces.shape[-1]/5
traces=traces-scipy.ndimage.filters.percentile_filter(traces,8,size=[num_sampls,1])

idx_components, fitness, erfc ,r_values, num_significant_samples = cse.utilities.evaluate_components(Y,traces,A,N=5,robust_std=True,thresh_finess=-10,compute_r_values=False)
sure_in_idx= idx_components[np.logical_and(fitness<-7.5,True)]

print ('Keeping ' + str(len(sure_in_idx)) + ' components out of ' + str(len(idx_components)))
#%%
save_results=True
if save_results:
    np.savez(os.path.join(os.path.split(fname_new)[0],'results_analysis.npz'),Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A.todense(),C2=C,b2=b,f2=f,YrA=YrA,sn=sn,d1=d1,d2=d2,idx_components=idx_components, fitness=fitness, erfc=erfc)    
     
#%%
pl.figure()
crd = cse.utilities.plot_contours(A.tocsc()[:,sure_in_idx],Cn,thr=0.9)
#%% get rid of evenrually noisy components. 
# But check by visual inspection to have a feeling fot the threshold. Try to be loose, you will be able to get rid of more of them later!
traces=C+YrA
idx_components, fitness, erfc,r_values,num_significant_samples = cse.utilities.evaluate_components(Y,traces,A,N=5,robust_std=False)

sure_in_idx= idx_components[np.logical_and(np.array(num_significant_samples)>1 ,np.array(r_values)>=.5)]
doubtful = idx_components[np.logical_and(np.array(num_significant_samples)==1 ,np.array(r_values)>=.5)]
they_suck = idx_components[np.logical_and(np.array(num_significant_samples)>=0 ,np.array(r_values)<.5)]
#%%
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A.tocsc()[:,sure_in_idx]),C[sure_in_idx,:],b,f, dims[0],dims[1], YrA=YrA[sure_in_idx,:],img=Cn)  
#%% visualize components
#pl.figure();
pl.subplot(1,3,1)
crd = cse.utilities.plot_contours(A.tocsc()[:,sure_in_idx],Cn,thr=0.9)
pl.subplot(1,3,2)
crd = cse.utilities.plot_contours(A.tocsc()[:,doubtful],Cn,thr=0.9)
pl.subplot(1,3,3)
crd = cse.utilities.plot_contours(A.tocsc()[:,they_suck],Cn,thr=0.9)
#%%
cse.utilities.stop_server(is_slurm = (backend == 'SLURM')) 
log_files=glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
