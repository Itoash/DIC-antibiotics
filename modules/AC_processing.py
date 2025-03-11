#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import cv2
from scipy.signal import detrend
from scipy.signal import argrelextrema
from cellpose_omni import io, transforms
from omnipose.utils import normalize99
from copy import deepcopy
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES
import time
from scipy import interpolate
from scipy.stats import linregress
from scipy.signal import find_peaks_cwt,detrend,butter,sosfiltfilt
from scipy.signal import peak_prominences,correlate,detrend,fftconvolve
from scipy.ndimage import convolve
import concurrent.futures
import matplotlib as mpl

def cdf(y):
    window = np.ones(10)
    window/=len(window)
    conv = np.convolve(y,window,mode='full')
    return conv

def applybandpass(series,framerate,frequencies = (0.1,6)):
    sos = butter(4,btype='bandpass',fs = framerate,Wn=frequencies,output='sos')
    filtereddata = sosfiltfilt(sos,series,axis=0)
    #plt.plot(np.linspace(0,series.shape[0]*framerate,series.shape[0]),np.mean(filtereddata,axis=(1,2)))
    # plt.title("filtered signal")
    # plt.show()
    return filtereddata
        
def plotpixels(coordslist,series,N,framerate):
    peakcount = 0
    while len(coordslist) < 10:
        x = np.random.randint(0,series.shape[1])
        y = np.random.randint(0,series.shape[2])
        if series[0,x,y] != 0:
            coordslist.append([x,y])
    fig, axs = plt.subplots(3)
    fig.set_figheight(30)
    fig.set_figwidth(15)
    peakcount = 0
    for i,c in enumerate(coordslist):
        pix = series[:,c[0],c[1]]
        axs[0].plot(np.linspace(0,N/framerate,N),pix,alpha = 0.3)
        pix_fft = np.fft.fft(pix)
        pix_fft = 2*np.abs(pix_fft[0:len(pix_fft)//2])/len(pix_fft)
        pix_freq = np.fft.fftfreq(len(pix_fft)*2,1/framerate)[0:(len(pix_fft)*2)//2]
        axs[1].plot(pix_freq[2:],pix_fft[2:],alpha = 0.3)
        axs[2].plot(pix_freq[2:-1],cdf(pix_fft[2:])[10:])
        maximum = argrelextrema(pix_fft,np.greater,order = 5)
        maxima = pix_freq[maximum][0]
        nopeaks = np.where(maxima>0.95)
        if maxima >0.6 and maxima<=1.2:
            peakcount+=1
            
        
    
    
    print(f"Found {peakcount}/30 pixels with peaks")
    plt.show()
    
def sortfiles(path,gettimes = False):
    files = [f for f in listdir(path) if  f!=".DS_Store" and ".txt" not in f and "DICTL" in f]
    keys = []
    for f in files:
        splitfile = f.split("_")
        time = splitfile[-1]
        ampm = time[-2:]
        timenumeral = int(time[:-3])
        if ampm == "PM" and timenumeral <100000:
            timenumeral+=120000
        keys.append(timenumeral)
        
    _,sortedfiles =  zip(*sorted(zip(keys, files)))
   
    sortedfiles = list(sortedfiles)
    sortedfiles = [path+"/"+f for f in sortedfiles]
    if not gettimes:
        return sortedfiles
    else:
        return sortedfiles,keys
    
        
def segmentDComni(imgs):
    # read images in a list!!!
    
    
    # print some info about the images.
    for i in imgs:
        print('Original image shape:',i.shape)
        print('data type:',i.dtype)
        print('data range: min {}, max {}\n'.format(i.min(),i.max()))
    nimg = len(imgs)
    print('\number of images:',nimg)
    
    print('\n')
    
            
    # invert images (16-bit) 
    imgs = [65535-im for im in imgs]
    # initialize model
    model_name = 'bact_phase_omni'
    model = models.CellposeModel( model_type=model_name)
    
    chans = [0,0] #this means segment based on first channel, no second channel 
    
    
    n = range(nimg) # segment all images in list
    
    # define parameters
    params = {'channels':chans, # always define this with the model
              'rescale': None, # upscale or downscale your images, None = no rescaling 
              'mask_threshold': -2, # erode or dilate masks with higher or lower values between -5 and 5 
              'flow_threshold': 0.4, # default is .4, but only needed if there are spurious masks to clean up; slows down output
              'transparency': False, # transparency in flow output
              'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
              'cluster': False, # use DBSCAN clustering
              'resample': True, # whether or not to run dynamics on rescaled grid or original grid 
              'verbose': False, # turn on if you want to see more output 
              'tile': True, # average the outputs from flipped (augmented) images; slower, usually not needed 
              'niter': None, # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
              'augment': True, # Can optionally rotate the image and average network outputs, usually not needed 
              'affinity_seg': True, # new feature, stay tuned...
             }
    
    tic = time.time() 
    masks, flows, styles = model.eval([imgs[i] for i in n],**params)
    
    net_time = time.time() - tic
    
    print('total segmentation time: {}s'.format(net_time))
    
    return masks

def obtainACarray(path,framerate = 16.7,tolerance = 0.2,frequency = 1,periods = 10,start = 150,end = 400,maskonlycells = True,interpolation = True,detr = True):
    onlyfiles = [f for f in listdir(path) if isfile(join(path,f)) and f !=".DS_Store" and ".tif" in f]
    name = onlyfiles[0]
    params = name.split('_')
    fps = [el for el in params if 'fps' in el][0]
    fps = fps[:-3]
    fps = fps.split('v')
    try:
        framerate = int(fps[0])+int(fps[1])/10
    except IndexError:
        framerate = int(fps[0])
    print(framerate)
    sorted_files = sorted(onlyfiles, key=lambda x: int(x[-8:-4]))
    images = np.empty(len(sorted_files), dtype=object)
    for n in range(0, len(sorted_files)):
      images[n] = cv2.imread( join(path,sorted_files[n]),-1 ) 
    dims = np.shape(images[0])
    images_copy = np.empty((len(images),dims[0],dims[1]))
    
    for i in range(len(images)):
        images_copy[i,:,:] = images[i]
    
    series = images_copy
    def makekernel(size):
        kernel = np.full((1,size,size),1/(size**2))
        return kernel
    # kernel = makekernel(5)
    
    # series = convolve(series,kernel)
    meanseries = np.mean(series,axis = (1,2))
    DCarray = np.mean(series[100:,:,:],axis = 0)
    
        
    
    
    N = len(images)
    
    
    

    
    # perioada de 1 Hz
    #nperiods = int(T*frequency)
    nperiods = periods
    
    newlastindex = round(nperiods*framerate)
    while(newlastindex+start>end):
        nperiods-=1
        newlastindex = round(nperiods*framerate)
    print("No of periods used: "+str(nperiods))
    
    end = newlastindex+start
    series = series[start:end,:,:]
    DCarray = np.mean(series[:,:,:],axis = 0)
    maskforseries = np.zeros_like(series[0,:,:])
    
    if maskonlycells:
        
        DCarray = cv2.medianBlur(DCarray.astype('uint16'),3)
        DCarray = normalize99(DCarray)
        
        
        
        print(f"Max value in DCarray = {np.max(DCarray)}")
        print(f"Min value in DCarray = {np.min(DCarray)}")
        
        mask = segmentDComni([DCarray])[0]
        maskforseries[mask!=0] = 1
        fig,ax = plt.subplots(2)
        ax[0].imshow(DCarray,cmap='Greys')
        ax[1].imshow(mask,cmap='rainbow')
        plt.show()
    N = series.shape[0]
    DCarray = np.mean(series[:,:,:],axis = 0)
    meanseries = np.mean(series,axis = (1,2))
    
    
  
        
    
    
    backsignal = np.asarray([5088.7274, 5094.0680, 5097.0630, 5099.2354, 5099.5657, 5098.4146, 5098.2139, 5098.8259, 5100.8851, 5102.5116, 5105.9490, 5107.5793, 5110.1275, 5111.3212, 5113.6733, 5114.5483, 5115.4386, 5116.6946, 5115.6866, 5114.1719, 5113.5036, 5110.8468, 5109.2284, 5108.0907, 5107.4564, 5109.6154, 5110.2407, 5111.7477, 5113.0459, 5115.2682, 5116.0112, 5116.9059, 5118.3320, 5119.3747, 5118.8786, 5118.5967, 5117.1301, 5115.7798, 5112.5056, 5111.2536, 5110.1608, 5110.0472, 5110.9313, 5110.9967, 5113.7029, 5114.9663, 5116.7325, 5118.5049, 5118.8671, 5120.2964, 5120.2680, 5120.3085, 5119.3148, 5116.0310, 5114.2452, 5111.8668, 5110.5533, 5110.1673, 5110.4193, 5112.1197, 5112.7115, 5114.9917, 5115.9846, 5116.9907, 5119.0784, 5118.3955, 5119.5296, 5120.0461, 5118.6723, 5117.6949, 5114.7440, 5112.8830, 5109.3033, 5108.5235, 5108.5007, 5109.5659, 5111.1072, 5111.3664, 5112.8172, 5115.0386, 5114.9649, 5117.1388, 5117.9167, 5118.6596, 5118.2296, 5117.1970, 5114.6667, 5112.6980, 5109.9042, 5107.8220, 5106.8283, 5107.1765, 5107.9073, 5109.3648, 5110.7652, 5112.6088, 5113.3117, 5115.8507, 5115.7498, 5116.2162, 5115.9799, 5116.2020, 5113.7787, 5112.1300, 5110.4307, 5107.4423, 5104.9046, 5102.9391, 5103.0644, 5104.2593, 5106.0592, 5106.7622, 5108.5079, 5109.9688, 5112.0465, 5113.6630, 5114.2912, 5114.4057, 5113.1234, 5112.1382, 5109.2238, 5106.2288, 5103.7581, 5104.1170, 5103.7770, 5105.1307, 5105.8649, 5106.6034, 5106.9067, 5108.0548, 5110.1199, 5112.2010, 5112.2132, 5113.0097, 5111.5605, 5110.9669, 5110.1435, 5108.4397, 5105.2157, 5103.0148, 5101.6779, 5100.6686, 5101.8906, 5104.2553, 5105.1521, 5105.9464, 5106.1933, 5107.3006, 5108.8508, 5111.1255, 5110.3188, 5108.8916, 5106.6711, 5105.0764, 5102.6398, 5100.6772, 5099.1201, 5098.6780, 5099.0737, 5099.6686, 5101.4396, 5102.7085, 5105.5817, 5107.2305, 5107.0008, 5108.8469, 5109.4256, 5109.6671, 5108.0925, 5106.5467, 5104.5287, 5101.7928, 5100.3518, 5099.4298, 5099.3667, 5100.7436, 5102.1566, 5102.4399, 5104.7152, 5105.5542, 5106.4283, 5107.8597, 5108.4730, 5108.8667, 5109.2711, 5107.1175, 5105.4296, 5102.8552, 5100.3562, 5099.1655, 5097.5851, 5098.2875, 5099.5671, 5099.9625, 5102.5853, 5103.5467, 5105.4141, 5105.4376, 5106.3557, 5107.3736, 5107.9172, 5107.0633, 5105.8497, 5104.2666, 5101.3106, 5098.6961, 5096.9863, 5097.7092, 5097.7979, 5099.4459, 5100.0414, 5100.7795, 5103.2921, 5105.1174, 5106.1976, 5106.5633, 5107.4449, 5107.1117, 5105.8141, 5103.9799, 5101.2819, 5099.8236, 5096.9565, 5096.3403, 5095.9309, 5097.3515, 5097.5602, 5099.5331, 5101.0596, 5102.6188, 5104.0927, 5104.5124, 5105.4517, 5105.1886, 5105.7880, 5103.9356, 5101.9930, 5099.7457, 5096.7448, 5095.0664, 5094.2146, 5093.9930, 5094.8464, 5096.5934, 5098.2375, 5100.9332, 5101.3679, 5102.9043, 5104.0118, 5103.6051, 5104.0376, 5103.6954, 5101.0347, 5098.1643, 5095.8322, 5093.9488, 5092.5304, 5093.2230, 5093.7054, 5095.1807, 5096.5532, 5098.0419, 5100.0342, 5101.5433, 5102.9384, 5103.7709, 5103.6751, 5102.7906, 5101.9107, 5100.2848, 5097.6618, 5095.2526, 5093.8186, 5092.4464, 5092.9240, 5093.5644, 5094.7981, 5095.8950, 5097.4236, 5098.6299, 5100.3516, 5100.8380, 5101.8932, 5102.0470, 5100.8524, 5099.8949, 5097.4077, 5095.1954, 5093.4638, 5091.6493, 5090.4229, 5091.1317, 5091.6314, 5093.5935, 5095.9171, 5096.7847, 5099.2280, 5100.2456, 5100.6367, 5102.6501, 5101.5124, 5100.5520, 5099.2912, 5097.5730, 5094.4440, 5092.1981, 5090.3884, 5089.8559, 5091.7391, 5092.4430, 5093.4040, 5095.2028, 5096.6363, 5097.0043, 5098.1133, 5098.4786, 5099.3434, 5099.5506, 5098.7699, 5096.1316, 5093.6036, 5091.5991, 5088.7406, 5088.1288, 5088.8933, 5089.0285, 5090.5634, 5092.0778, 5094.4157, 5095.3772, 5097.8693, 5098.3009, 5099.4259, 5099.2872, 5098.3486, 5096.9209, 5095.0601, 5091.8648, 5090.1533, 5088.2476, 5087.3696, 5088.0443, 5088.9176, 5090.1406, 5091.2349, 5093.3112, 5094.6794, 5096.3078, 5096.9453, 5097.5402, 5096.9879, 5096.2135, 5094.4734, 5091.3374, 5089.8143, 5088.0756, 5086.1096, 5086.4507, 5087.7977, 5088.6876, 5089.6621, 5092.3885, 5092.3612, 5094.2242, 5094.9551, 5095.6320, 5095.6283, 5096.2693, 5094.7985, 5092.4927, 5088.9570, 5086.8063, 5085.4455, 5084.8714, 5084.1705, 5085.3968, 5087.5300, 5088.6004, 5090.1299, 5091.5108, 5092.8117, 5094.8066, 5094.8801, 5094.9655, 5094.5526, 5093.8367, 5090.7430, 5087.9250, 5085.3970, 5084.4204, 5083.9557, 5085.1557, 5085.2973, 5086.8909, 5088.3842, 5090.2434, 5090.5722, 5092.6598, 5093.9276, 5094.3696])
    backsignal = backsignal[start:end]
    if interpolation:
        
        currentinterval_s = 1/framerate
        desiredinterval_s = 1/np.round(framerate)
        N = series.shape[0]
        print(N*currentinterval_s)
        newN = round(N*currentinterval_s/desiredinterval_s)
        interpseries = np.zeros((newN,series.shape[1],series.shape[2]))
        synthetic_t = np.linspace(0,(newN)*desiredinterval_s,newN)
        real_t = np.linspace(0,N*currentinterval_s,N)
        spl = interpolate.PchipInterpolator(real_t, series,axis = 0)
        backspl = interpolate.PchipInterpolator(real_t,backsignal)
        backsignal = backspl(synthetic_t)
        interpseries = spl(synthetic_t)
        print(f"Changed frame interval to {desiredinterval_s},total time is {newN*desiredinterval_s}")
        series = interpseries
        framerate = 1/desiredinterval_s
        
        
    #plotpixels([(198,198)],series,series.shape[0],framerate)
    meanseriesforlater = np.mean(series,axis = (1,2))
    t = np.linspace(0,len(meanseriesforlater)/framerate,len(meanseriesforlater))
    maskedseries = deepcopy(series)
    for i in range(maskedseries.shape[0]):
        k = maskedseries[i,:,:]
        k[maskforseries==0]=0
        
    def convolveperpix(arr,framerate = framerate):
        kernel = np.ones(round(framerate))/round(framerate)
        kernel = kernel[:,np.newaxis,np.newaxis]
        convolvedseries = fftconvolve(arr,kernel,mode='same',axes = 0)
        return convolvedseries
        
    if detr:
        
        series_dperpix = detrend(series,axis = 0)
        maskedseries = detrend(maskedseries,axis=0)
        backsignal = detrend(backsignal)
        coeffs = np.polyfit(t,np.mean(series,axis = (1,2)),deg = 1)
        trend = coeffs[0]*t+coeffs[1]
        series_dpermean = series-trend[:,np.newaxis,np.newaxis]
        series = series_dperpix
        
    
    filteredsignal = applybandpass(series, framerate)
    backsignal = applybandpass(backsignal,framerate)
    series = filteredsignal
    
   
    
    
    series_fft = np.fft.fft(series,axis = 0)
    nfft = len(series_fft)
    fftcopy = deepcopy(series_fft)
    series_fft = 2*np.abs(series_fft)[0:nfft//2]
    series_fft /= N
    series_frequencies = np.fft.fftfreq(nfft,1/framerate)[0:nfft//2]
    plt.plot(series_frequencies[2:],np.mean(series_fft,axis = (1,2))[2:])
    plt.show()
    # [print(fq,ff) for fq,ff in zip(series_frequencies,np.mean(series_fft,axis = (1,2)))]
    good_freq=series_frequencies.flat[np.abs(series_frequencies - frequency).argmin()]
    good_freq = np.argwhere(series_frequencies==good_freq)[0]
    chosenfrequency = series_frequencies[good_freq]
    
    print(chosenfrequency)
   
    
  
    ACarray = np.mean(series_fft[good_freq,:,:],axis = 0)
    
    def computecorrelation(signalmatrix,reference):
        if signalmatrix.shape[0] != reference.shape[0]:
            print("Different lengths for signals!")
            return signalmatrix,np.mean(signalmatrix,axis=(1,2))
        if len(reference.shape)==1:
            corrmat = np.sum(np.multiply(signalmatrix,reference[:,np.newaxis,np.newaxis]),axis = 0)
        else:
            corrmat = np.sum(np.multiply(signalmatrix,reference),axis = 0)
        
        corrint = np.sum(signalmatrix**2,axis = 0)
        corrint[corrint==0]=1
        bckgint = np.sum(reference**2)
        if bckgint == 0:
            bckgint =1
        corrmat /= np.sqrt(corrint*bckgint)
        signalcorr = [] 
        signalcorr.append(np.mean(corrmat[:25,:25]))
        signalcorr.append(np.mean(corrmat[-25:,:25]))
        signalcorr.append(np.mean(corrmat[:25,-25:]))
        signalcorr.append(np.mean(corrmat[-25:,-25:])) 
        print(f"corrmatrix max is {np.max(corrmat)}")
        print(f"corrmatrix min is {np.min(corrmat)}")
        corrmat +=1
        
        return corrmat,signalcorr
   
    
    
    
    
    
    # background = np.mean(series[:,:,:],axis = (1,2))
    # corrmatrix,signalcorr = computecorrelation(series[:,:,:],background)
         
    t = np.linspace(0,series.shape[0]/framerate,series.shape[0])
    meanfft = np.fft.fft(np.mean(series[:,:50,50:],axis=(1,2)))
    angle = np.angle(meanfft[good_freq])
    background = np.cos(2*np.pi*frequency*t+angle)    
    background = series[:,:100,:100]
    for i in range(background.shape[0]):
        k = background[i,:,:]
        med = np.mean(k)
        percent = np.percentile(ACarray,99.5)
        k[ACarray[:100,:100]>=percent] = med
        background[i,:,:] = k
    background = np.mean(background,axis = (1,2))
    
    synthcorrmatrix,synthcorrs = computecorrelation(series,background)
    
    
    # background = np.mean(series[:,:,:],axis = (1,2))
    # maskedcorr,maskedsignals = computecorrelation(maskedseries,background)
    
    # maskedcorr = np.nanmean(maskedcorr,axis = (0,1))
    
   
   
 
   
    
    plt.hist(synthcorrmatrix.flatten(),bins = 200)
    plt.show()
    
    plt.imshow(ACarray)
    
    plt.show()
    plt.imshow(synthcorrmatrix,cmap='gray')
    plt.show()
    corrimage = synthcorrmatrix
    return ACarray,DCarray,corrimage,synthcorrmatrix,meanseriesforlater,mask

def main(mypath = "/Users/victorionescu/electro-optic/16 jan 25 CTRL"):
    
    
    sorted_filenames = sortfiles(mypath)
    
    signals = []
    images = []
    fullcorrs = []
    cellcorrs = []
    proms = []
    synthimgs = []
    DCs = []
    
    
    limits,ps = testsignals(mypath)
    timestrings,exptimes = gettimes(mypath)
    print(len(sorted_filenames))
    for i,f in enumerate(sorted_filenames):
        print("------")
        print(f"Image no {i+1}")
        print("-----\n")
        image,DC,promimg,synthimg,signal,mask = obtainACarray(f,frequency=1,start = limits[i][0],end = limits[i][1],periods = min(ps))
        
                     
        proms.append(promimg)
        synthimgs.append(synthimg)
        promimg = promimg*10000
        promimg = np.clip(promimg,0,65535)
        promimg = promimg.astype('uint16')
        DCs.append(DC)
        cv2.imwrite('proms-'+str(timestrings[i])+'.tif',promimg)
        
        image = image*1000
        image = image.astype('uint16')
        images.append(image)
        image=np.clip(image,0,65535)
        cv2.imwrite('AC-'+str(timestrings[i])+'.tif',image)
        
        DC = DC.astype('uint16')
        cv2.imwrite('DC-'+str(timestrings[i])+'.tif',DC)
        
        signals.append(signal)
        
        mask = mask.astype('uint16')
        cv2.imwrite('Omnimask-'+str(timestrings[i])+'.tif',mask)
        
    return images,DCs,proms,synthimgs,signals,fullcorrs,cellcorrs

def testperiods(mypath = "/Users/victorionescu/Desktop/6dec",limits = (0,-1),periods = (3,16)):
    filenames = [mypath+"/"+f for f in listdir(mypath) if f !=".DS_Store" ]
    sorted_filenames = sorted(filenames, key=lambda x: x.split('_')[-1][:-3])
    ACarrays = []
    signals = []
    for k in range(periods[0],periods[1]):
        ACs = []
        sigs = []
        for i in range(limits[0],limits[1]):
            image,DC,promimg,synthimg,signal,signalcorr,cellcorr = obtainACarray(sorted_filenames[i],periods = k)
            ACs.append(image)
            sigs.append(signal)
        ACarrays.append(ACs)
        signals.append(sigs)
    return ACarrays,signals

def teststarts(mypath = "/Users/victorionescu/Desktop/6dec",startindexes = [150]):
    filenames = [mypath+"/"+f for f in listdir(mypath) if f !=".DS_Store" ]
    sorted_filenames = sorted(filenames, key=lambda x: x.split('_')[-1][:-3])
    ACarrays = []
    signals = []
    for k in startindexes:
        ACs = []
        sigs = []
        for i in range(len(sorted_filenames)):
            image,DC,promimg,synthimg,signal,signalcorr,cellcorr = obtainACarray(sorted_filenames[i],start = k)
            ACs.append(image)
            sigs.append(signal)
        ACarrays.append(ACs)
        signals.append(sigs)
    return ACarrays,signals

def plotsignals(mypath = "/Users/victorionescu/Desktop/6dec"):
    sorted_filenames = sortfiles(mypath)
    for j,s in enumerate(sorted_filenames):
        onlyfiles = [f for f in listdir(s) if f !=".DS_Store" and ".txt" not in f]
        name = onlyfiles[0]
        params = name.split('_')
        fps = [el for el in params if 'fps' in el][0]
        fps = fps[:-3]
        fps = fps.split('v')
        framerate = int(fps[0])+int(fps[1])/10
        print(framerate)
        sorted_files = sorted(onlyfiles, key=lambda x: int(x[-8:-4]))
        images = np.empty(len(sorted_files), dtype=object)
        for n in range(0, len(sorted_files)):
          images[n] = cv2.imread( join(s,sorted_files[n]),-1 ) 
        dims = np.shape(images[0])
        images_copy = np.empty((len(images),dims[0],dims[1]))
        
        for i in range(len(images)):
            images_copy[i,:,:] = images[i]
        
        series = images_copy
        
        meanseries = np.mean(series,axis = (1,2))
        print(np.argwhere(meanseries == 0))
        kernel = np.ones(round(framerate))/round(framerate)
        kernel = kernel[:,np.newaxis,np.newaxis]
        convolvedseries = fftconvolve(series,kernel,mode='valid',axes = 0)
        convolvedseries = np.mean(convolvedseries,axis = (1,2))
        fig,ax = plt.subplots(3)
        ax[0].plot(np.linspace(0,len(meanseries)/framerate,len(meanseries)),meanseries)
        ax[0].scatter(np.linspace(0,len(meanseries)/framerate,len(meanseries)),meanseries,marker = '*')
        ax[1].plot(np.linspace(0,len(convolvedseries)/framerate,len(convolvedseries)),convolvedseries)
        ax[1].scatter(np.linspace(0,len(convolvedseries)/framerate,len(convolvedseries)),convolvedseries,marker = '*')
        ax[2].plot(np.linspace(0,len(convolvedseries)/framerate,len(convolvedseries)),meanseries[round(framerate)//2:-round(framerate)//2+1]-convolvedseries)
        ax[2].scatter(np.linspace(0,len(convolvedseries)/framerate,len(convolvedseries)),meanseries[round(framerate)//2:-round(framerate)//2+1]-convolvedseries,marker = '*')
        
        fig.suptitle("Image no. "+str(j))

        plt.show()
        
def testsignals(mypath = "/Users/victorionescu/Desktop/6dec"):
    sorted_filenames = sortfiles(mypath)
    limits = []
    periods = []
    for j,s in enumerate(sorted_filenames):
        onlyfiles = [f for f in listdir(s) if f !=".DS_Store" and ".tif" in f]
        name = onlyfiles[0]
        params = name.split('_')
        fps = [el for el in params if 'fps' in el][0]
        fps = fps[:-3]
        
        fps = fps.split('v')
        try:
            framerate = int(fps[0])+int(fps[1])/10
        except IndexError:
            framerate = int(fps[0])
        sorted_files = sorted(onlyfiles, key=lambda x: int(x[-8:-4]))
        images = np.empty(len(sorted_files), dtype=object)
        for n in range(0, len(sorted_files)):
          images[n] = cv2.imread( join(s,sorted_files[n]),-1 ) 
        dims = np.shape(images[0])
        images_copy = np.empty((len(images),dims[0],dims[1]))
        
        for i in range(len(images)):
            images_copy[i,:,:] = images[i]
        
        series = images_copy
        
        meanseries = np.mean(series,axis = (1,2))
        
        defaults = (150,399)
        
        diff = np.diff(meanseries)
        outliers = np.argwhere(abs(diff)>np.std(meanseries)).flatten()
        outliers = np.append(outliers,defaults)
        outliers = np.sort(outliers)
        outlierdiffs = np.diff(outliers)
        selectedrange = (outliers[np.argmax(outlierdiffs)]+1,outliers[np.argmax(outlierdiffs)+1]-1)
        time = np.linspace(0,len(meanseries)/framerate,len(meanseries))
        meanseries = meanseries[selectedrange[0]:selectedrange[1]]
        time = time[selectedrange[0]:selectedrange[1]]
        meanseries = detrend(meanseries)
        meanfft = np.fft.fft(meanseries)
        meanfft = np.abs(meanfft[0:len(meanfft)//2])*2/len(meanfft)
        meanfreq = np.fft.fftfreq(len(meanfft)*2,1/framerate)[0:len(meanfft)]
       
        nperiods = (selectedrange[1]-selectedrange[0])/framerate
        nperiods = np.floor(nperiods)
        limits.append(selectedrange)
        periods.append(nperiods)
       
        
    return limits,periods
        

def gettimes(mypath = "/Users/victorionescu/Desktop/6dec"):
    _,times = sortfiles(mypath,gettimes = True)
    times = list(sorted(times))
    times = [x//100 for x in times]
    hour = [x//100 for x in times]
    minute = [x%100 for x in times]
    timestrings = [str(h)+"_"+str(m) for h,m in zip(hour,minute)]
    hour = [x*60 for x in hour]
    
    exptimes = [h+m for h,m in zip(hour,minute)]
    exptimes = [e-exptimes[0]for e in exptimes]
    
    
    return timestrings,exptimes
    
def testcorr(path,framerate = 16.7,tolerance = 0.2,frequency = 1,periods = 10,start = 150,end = 400,maskonlycells = True,interpolation = True,detr = True):
    onlyfiles = [f for f in listdir(path) if isfile(join(path,f)) and f !=".DS_Store" and ".tif" in f]
    name = onlyfiles[0]
    params = name.split('_')
    fps = [el for el in params if 'fps' in el][0]
    fps = fps[:-3]
    fps = fps.split('v')
    try:
        framerate = int(fps[0])+int(fps[1])/10
    except IndexError:
        framerate = int(fps[0])
    print(framerate)
    sorted_files = sorted(onlyfiles, key=lambda x: int(x[-8:-4]))
    images = np.empty(len(sorted_files), dtype=object)
    for n in range(0, len(sorted_files)):
      images[n] = cv2.imread( join(path,sorted_files[n]),-1 ) 
    dims = np.shape(images[0])
    images_copy = np.empty((len(images),dims[0],dims[1]))
    
    for i in range(len(images)):
        images_copy[i,:,:] = images[i]
    
    series = images_copy
    def makekernel(size):
        kernel = np.full((1,size,size),1/(size**2))
        return kernel
    # kernel = makekernel(5)
    
    # series = convolve(series,kernel)
    meanseries = np.mean(series,axis = (1,2))
    DCarray = np.mean(series[100:,:,:],axis = 0)
    
        
    
    
    N = len(images)
    
    
    

    
    # perioada de 1 Hz
    #nperiods = int(T*frequency)
    nperiods = periods
    
    newlastindex = round(nperiods*framerate)
    while(newlastindex+start>end):
        nperiods-=1
        newlastindex = round(nperiods*framerate)
    print("No of periods used: "+str(nperiods))
    
    end = newlastindex+start
    series = series[start:end,:,:]
    DCarray = np.mean(series[:,:,:],axis = 0)
    maskforseries = np.zeros_like(series[0,:,:])
    
    if maskonlycells:
        
        DCarray = cv2.medianBlur(DCarray.astype('uint16'),3)
        DCarray = normalize99(DCarray)
        
        
        
        print(f"Max value in DCarray = {np.max(DCarray)}")
        print(f"Min value in DCarray = {np.min(DCarray)}")
        
        mask = segmentDComni([DCarray])[0]
        maskforseries[mask!=0] = 1
        fig,ax = plt.subplots(2)
        ax[0].imshow(DCarray,cmap='Greys')
        ax[1].imshow(mask,cmap='rainbow')
        plt.show()
    N = series.shape[0]
    DCarray = np.mean(series[:,:,:],axis = 0)
    meanseries = np.mean(series,axis = (1,2))
    back_mean = meanseries
    selseries=  series[:,:100,:100]
    lowerone = np.percentile(selseries,1,axis = (1,2))
    upperone = np.percentile(selseries,99,axis = (1,2))

    for i in range(selseries.shape[0]):
        k = selseries[i,:,:]
        med = np.mean(k)
        
        k[k<lowerone[i]] = med
        k[k>upperone[i]] = med
        selseries[i,:,:] = k
    
    back_filtered = np.mean(selseries,axis = (1,2))
    back_selection = np.mean(series[:,:100,:100],axis = (1,2))
    back_segment = np.mean(series,axis=(1,2),where=maskforseries!=0)
    
  
        
    
    
    
    if interpolation:
        
        currentinterval_s = 1/framerate
        desiredinterval_s = 1/np.round(framerate)
        N = series.shape[0]
        print(N*currentinterval_s)
        newN = round(N*currentinterval_s/desiredinterval_s)
        interpseries = np.zeros((newN,series.shape[1],series.shape[2]))
        synthetic_t = np.linspace(0,(newN)*desiredinterval_s,newN)
        real_t = np.linspace(0,N*currentinterval_s,N)
        spl = interpolate.PchipInterpolator(real_t, series,axis = 0)
        backspl = interpolate.PchipInterpolator(real_t,back_mean)
        back_mean = backspl(synthetic_t)
        backspl = interpolate.PchipInterpolator(real_t,back_selection)
        back_selection = backspl(synthetic_t)
        backspl = interpolate.PchipInterpolator(real_t,back_segment)
        
        back_segment = backspl(synthetic_t)
        backspl = interpolate.PchipInterpolator(real_t,back_filtered)
        back_filtered = backspl(synthetic_t)
        interpseries = spl(synthetic_t)
        print(f"Changed frame interval to {desiredinterval_s},total time is {newN*desiredinterval_s}")
        series = interpseries
        framerate = 1/desiredinterval_s
        
        
    #plotpixels([(198,198)],series,series.shape[0],framerate)
    meanseriesforlater = np.mean(series,axis = (1,2))
    t = np.linspace(0,len(meanseriesforlater)/framerate,len(meanseriesforlater))
    maskedseries = deepcopy(series)
    for i in range(maskedseries.shape[0]):
        k = maskedseries[i,:,:]
        k[maskforseries==0]=0
        
    
    if detr:
        
        series_dperpix = detrend(series,axis = 0)
        maskedseries = detrend(maskedseries,axis=0)
        back_mean = detrend(back_mean)
        back_selection = detrend(back_selection)
        back_segment = detrend(back_segment)
        back_filtered = detrend(back_filtered)
        coeffs = np.polyfit(t,np.mean(series,axis = (1,2)),deg = 1)
        trend = coeffs[0]*t+coeffs[1]
        series_dpermean = series-trend[:,np.newaxis,np.newaxis]
        series = series_dperpix
        
    
    filteredsignal = applybandpass(series, framerate)
    back_mean = applybandpass(back_mean,framerate)
    back_selection = applybandpass(back_selection,framerate)
    back_segment = applybandpass(back_segment,framerate)
    back_filtered = applybandpass(back_filtered,framerate)
    series = filteredsignal
    
   
    
    
    series_fft = np.fft.fft(series,axis = 0)
    nfft = len(series_fft)
    fftcopy = deepcopy(series_fft)
    series_fft = 2*np.abs(series_fft)[0:nfft//2]
    series_fft /= N
    series_frequencies = np.fft.fftfreq(nfft,1/framerate)[0:nfft//2]
    plt.plot(series_frequencies[2:],np.mean(series_fft,axis = (1,2))[2:])
    plt.show()
    # [print(fq,ff) for fq,ff in zip(series_frequencies,np.mean(series_fft,axis = (1,2)))]
    good_freq=series_frequencies.flat[np.abs(series_frequencies - frequency).argmin()]
    good_freq = np.argwhere(series_frequencies==good_freq)[0]
    chosenfrequency = series_frequencies[good_freq]
    
    print(chosenfrequency)
   
    
  
    ACarray = np.mean(series_fft[good_freq,:,:],axis = 0)
    plt.plot(synthetic_t,np.mean(series,axis = (1,2)))
    def computecorrelation(signalmatrix,reference):
        if signalmatrix.shape[0] != reference.shape[0]:
            print("Different lengths for signals!")
            return signalmatrix,np.mean(signalmatrix,axis=(1,2))
        if len(reference.shape)==1:
            corrmat = np.sum(np.multiply(signalmatrix,reference[:,np.newaxis,np.newaxis]),axis = 0)
        else:
            corrmat = np.sum(np.multiply(signalmatrix,reference),axis = 0)
        
        corrint = np.sum(signalmatrix**2,axis = 0)
        corrint[corrint==0]=1
        bckgint = np.sum(reference**2)
        if bckgint == 0:
            bckgint =1
        corrmat /= np.sqrt(corrint*bckgint)
        signalcorr = [] 
        signalcorr.append(np.mean(corrmat[:25,:25]))
        signalcorr.append(np.mean(corrmat[-25:,:25]))
        signalcorr.append(np.mean(corrmat[:25,-25:]))
        signalcorr.append(np.mean(corrmat[-25:,-25:])) 
        print(f"corrmatrix max is {np.max(corrmat)}")
        print(f"corrmatrix min is {np.min(corrmat)}")
        corrmat +=1
        
        return corrmat,signalcorr
   
    
    
    
    
    
    # background = np.mean(series[:,:,:],axis = (1,2))
    # corrmatrix,signalcorr = computecorrelation(series[:,:,:],background)
         
    t = np.linspace(0,series.shape[0]/framerate,series.shape[0])
    meanfft = np.fft.fft(np.mean(series[:,:50,50:],axis=(1,2)))
    angle = np.angle(meanfft[good_freq])
    background = np.cos(2*np.pi*frequency*t+angle) 
    background = series[:,:100,:100]
    for i in range(background.shape[0]):
        k = background[i,:,:]
        med = np.mean(k)
        percent = np.percentile(ACarray,99.5)
        k[ACarray[:100,:100]>=percent] = med
        background[i,:,:] = k
    background = np.mean(background,axis = (1,2))
    synthcorrmatrix,synthcorrs = computecorrelation(series,background)
    meancorrmatrix,meansig = computecorrelation(series,back_mean)
    segcorrmatrix,segsig = computecorrelation(series,back_segment)
    selcorrmatrix,selsig = computecorrelation(series,back_selection)
    filtcorrmatrix,filtsig = computecorrelation(series,back_filtered)
    
    # background = np.mean(series[:,:,:],axis = (1,2))
    # maskedcorr,maskedsignals = computecorrelation(maskedseries,background)
    
    # maskedcorr = np.nanmean(maskedcorr,axis = (0,1))
   
   
 
   
    
    # plt.hist(synthcorrmatrix.flatten(),bins = 200)
    # plt.show()
    
    # plt.imshow(ACarray)
    
    # plt.show()
    # plt.imshow(synthcorrmatrix,cmap='gray')
    # plt.show()
    corrimage = synthcorrmatrix
    return background,back_mean,back_selection,back_segment,back_filtered
    
def testallcorrs(path):
    sorted_filenames = sortfiles(path)
    synth =[]
    mean = []
    select = []
    segment = []
    filtered = []
    for i,f in enumerate(sorted_filenames):
        sy,me,sel,seg,filt = testcorr(f)
        synth.append(sy)
        mean.append(me)
        select.append(sel)
        segment.append(seg)
        filtered.append(filt)
        for el,name in zip([sy,me,seg,sel,filt],["Synth","Mean sig","Segment sig","Selection sig","Filtered sig"]):
            el = el*10000
            el = np.clip(el,0,65535)
            el = el.astype('uint16')
            cv2.imwrite(name+"-"+str(i)+".tif",el)
       

    return synth,mean,select,segment

def probedistributions(expname,folder = '/Users/victorionescu/electro-optical/'):
    
    fullname = folder+expname
    sorted_filenames = sortfiles(fullname)
    sorted_filenames = sorted_filenames[:10]
    cmap = mpl.colormaps['viridis'].resampled(len(sorted_filenames))
    for j,file in enumerate(sorted_filenames):
  
        onlyfiles = [f for f in listdir(file) if isfile(file+"/"+f) and f !=".DS_Store" and ".tif" in f]
 
        sorted_files = sorted(onlyfiles, key=lambda x: int(x[-8:-4]))
        
        images = np.empty(len(sorted_files), dtype=object)
        for n in range(0, len(sorted_files)):
            images[n] = cv2.imread( join(file+"/",sorted_files[n]),-1 )
        dims = np.shape(images[0])
        images_copy = np.empty((len(images),dims[0],dims[1]))
        
        for i in range(len(images)):
            images_copy[i,:,:] = images[i]
        images = images_copy
        
        images = np.reshape(images,images.shape[0]*images.shape[1]*images.shape[2])
        
        plt.hist(images,histtype='step',bins=100,color = cmap(j),label = [n for n in file.split('/') if 'DICT' in n][0][44:],density=True)
        
    plt.xlim(3000,6000)
    plt.title(expname)
    plt.legend()
    plt.show()
  
        
        


#%%   
explist = ['18 dec 24 CST', '4 nov 24 CTRL', '18 feb 25 CST1', '11 dec 24 CST2', '25 februarie_5 mgL CST la 12 si 32',  '10 dec 24 CST', 
 '5 nov 24 CST', '19 feb 25 CST2', '9 dec 24 CST', '25 februarie_2v5 mgL CST la 16 si 33', '3 martie_1v25 CST la 1 si 17', 
 '20 feb 25 CST1', '12 dec 24 CST', '17 feb 25 CST', '18 feb 25 CST2', '26 februarie_2v5 mgL CST la 12 si 30', '11 dec 24 CST1',
 '19 dec 24 CST', '12 nov 24 CST', '6 dec 24 CTRL', '19 feb 25 CST1', '6 nov 24 CTRL', '16 jan 25 CTRL', '17 jan 25 CST', '20 feb 25 CST2',
 '24 feb 25 CST', '5 dec 24 CST', '12 jul 24 CST']
explist = ['19 feb 25 CST1','19 feb 25 CST2','20 feb 25 CST1','20 feb 25 CST2','24 feb 25 CST']

for e in explist:
    
    probedistributions(e)
#%%
plotsignals("/Users/victorionescu/electro-optical/20 feb 25 CST2")

#%%
synth,mean,select,segment = testallcorrs("/Users/victorionescu/electro-optical/20 feb 25 CST2")
#%%
for s,f in zip(synth,select):
    plt.plot(np.linspace(0,len(s),len(s)),s,label = 'cu outlier removal')
    plt.plot(np.linspace(0,len(f),len(f)),f,label = 'fara outlier removal')
    plt.show()
#%%
if __name__ == '__main__':
    imgs,DCs,proms,synthcorrs,signals,signalcorrs,cellcorrs = main(mypath = "/Users/victorionescu/electro-optical/20 feb 25 CST2")
    
#%%
testsignals('/Users/victorionescu/electro-optical/24 feb 25 CST')
#%%
for s in signals:
    plt.plot(np.linspace(0,len(s),len(s)),s)
plt.show()
for i in range(4):
    plt.plot(np.linspace(0,len(signalcorrs)*3,len(signalcorrs)),[np.arccos(s[i]) for s in signalcorrs])

plt.plot(np.linspace(0,len(cellcorrs)*3,len(cellcorrs)),[np.arccos(el) for el in cellcorrs],c = 'k')
plt.show()
#%%

# Acimages12nov3,signallist12nov3= testperiods(mypath = "/Users/victorionescu/Desktop/12 noiembrie-imagini raw",periods=(2,12),limits= (0,50))
# Acimages6nov3,signallist6nov3= testperiods(mypath = "/Users/victorionescu/Desktop/6noi24_CTRL",periods=(2,12),limits = (0,32))

# Acimages6dec3,signallist6dec3= testperiods(mypath = "/Users/victorionescu/Desktop/6dec",periods=(2,16),limits = (0,32))
#Acimages17ian3,signallist17ian3= testperiods(mypath = "/Users/victorionescu/Desktop/17 ian 25_2v5CST",periods=(2,12),limits = (0,50))
#Acimages10dec3,signallist10dec3= testperiods(mypath = "/Users/victorionescu/Desktop/10 decembrie 24",periods=(2,12),limits = (0,69))
# Acimages9dec3,signallist9dec3= testperiods(mypath = "/Users/victorionescu/Desktop/9 decembrie 24",periods=(2,12),limits = (0,42))
# Acimages16ian3,signallist16ian3= testperiods(mypath = "/Users/victorionescu/Desktop/16 ianuarie 25",periods=(2,12),limits = (0,44))
#%%
# Acimages12nov,signallist12nov= teststarts(mypath = "/Users/victorionescu/Desktop/12 noiembrie-imagini raw",startindexes=[150,175,200,250])
# Acimages6nov,signallist6nov= teststarts(mypath = "/Users/victorionescu/Desktop/6noi24_CTRL",startindexes=[150,175,200,250])

# Acimages6dec,signallist6dec= teststarts(mypath = "/Users/victorionescu/Desktop/6dec",startindexes=[150,175,200,250])

# Acimages17ian,signallist17ian = teststarts(mypath = "/Users/victorionescu/Desktop/17 ian 25_2v5CST",startindexes=[150,175,200,250])
#%%
plotsignals("/Users/victorionescu/electro-optical/5 dec 24 CST")


#%%
Ac6nov_means = [[ np.mean(im) for im in x] for x in Acimages6nov3]
Ac12nov_means = [[ np.mean(im) for im in x] for x in Acimages12nov3]
Ac6dec_means = [[np.mean(im) for im in x] for x in Acimages6dec3]
Ac17ian_means = [[np.mean(im) for im in x] for x in Acimages17ian3]
Ac9dec_means = [[np.mean(im) for im in x] for x in Acimages9dec3]
Ac16ian_means = [[np.mean(im) for im in x] for x in Acimages16ian3]
Ac10dec_means= [[np.mean(im) for im in x] for x in Acimages10dec3]

def computediffnestedlist(lis):
    diffarray = []
    for i in range(len(lis)-1):
        now = np.asarray(lis[i])
        nex = np.asarray(lis[i+1])
        diff = np.sum(nex-now)
        diffarray.append(diff)
    return diffarray
nov6diff = computediffnestedlist(Ac6nov_means)
nov12diff = computediffnestedlist(Ac12nov_means)
dec6diff = computediffnestedlist(Ac6dec_means)
ian17diff = computediffnestedlist(Ac17ian_means)

viridis = mpl.colormaps['rainbow'].resampled(len(Ac6nov_means))
# for i,el in enumerate(Ac6nov_means):
#     plt.plot(np.linspace(0,len(el),len(el)),el,label = "#T: "+str(i+2),c = viridis(i))
# plt.legend(fancybox=True)
# plt.title("6 nov ctrl")
# plt.xlabel("# Img")
# plt.ylabel("Amplitudinea medie")
# plt.show()
# viridis = mpl.colormaps['rainbow'].resampled(len(Ac6dec_means))
# for i,el in enumerate(Ac6dec_means):
#     plt.plot(np.linspace(0,len(el),len(el)),el,label = "#T: "+str(i+2),c = viridis(i))
# plt.legend(fancybox=True)
# plt.title("6 dec ctrl")
# plt.xlabel("# Img")
# plt.ylabel("Amplitudinea medie")
# plt.show()
# viridis = mpl.colormaps['rainbow'].resampled(len(Ac12nov_means))
# for i,el in enumerate(Ac12nov_means):
#     plt.plot(np.linspace(0,len(el),len(el)),el,label = "#T: "+str(i+2),c = viridis(i))
# plt.legend(fancybox=True)
# plt.title("12 nov 10mgL")
# plt.xlabel("# Img")
# plt.ylabel("Amplitudinea medie")
# plt.show()

for i,el in enumerate(Ac17ian_means):
    plt.plot(np.linspace(0,len(el),len(el)),el,label = "#T: "+str(i+2),c = viridis(i))
plt.legend(fancybox=True)
plt.title("17 ian 2.5 mgL")
plt.xlabel("# Img")
plt.ylabel("Amplitudinea medie")
plt.show()

# for i,el in enumerate(Ac16ian_means):
#     plt.plot(np.linspace(0,len(el)-2,len(el))-2,[x if x<50 else None for x in el ],label = "#T: "+str(i+2),c = viridis(i))
# plt.legend(fancybox=True)
# plt.title("16 ian")
# plt.xlabel("# Img")
# plt.ylabel("Amplitudinea medie")
# plt.show()

for i,el in enumerate(Ac9dec_means):
    plt.plot(np.linspace(0,len(el),len(el)),el,label = "#T: "+str(i+2),c = viridis(i))
plt.legend(fancybox=True)
plt.title("9 dec")
plt.xlabel("# Img")
plt.ylabel("Amplitudinea medie")
plt.show()

for i,el in enumerate(Ac10dec_means):
    plt.plot(np.linspace(0,len(el[:-5]),len(el[:-5])),el[:-5],label = "#T: "+str(i+2),c = viridis(i))
plt.legend(fancybox=True)
plt.title("10 dec")
plt.xlabel("# Img")
plt.ylabel("Amplitudinea medie")
plt.show()

#%%
print(nov6diff)
plt.plot(np.linspace(2,12,9),nov6diff,label = '6 nov')
plt.plot(np.linspace(2,12,9),nov12diff,label = '12 nov')
plt.plot(np.linspace(2,12,9),ian17diff,label = '17 ian')
plt.plot(np.linspace(2,16,13),dec6diff, label = '6 dec')
plt.legend()
plt.title("Diferentele intre medii vs. nr de perioade")
plt.xlabel("# T")
plt.ylabel("Diferenta intre medii")


    
#%%
signallist = signallist12nov
viridis = mpl.colormaps['rainbow'].resampled(len(signallist[0]))
avgmax=[]
for i,el in enumerate(signallist[0]):
    elfreq = np.fft.fftfreq(len(el),0.05)[0:len(el)//2]
    elfft = 2*np.abs(np.fft.fft([x-np.mean(el) for x in el])[0:len(el)//2])/len(el)
    #plt.plot(np.linspace(0,len(el),len(el)),[x-np.mean(el) for x in el],c = viridis(i))
    plt.plot(elfreq,elfft)
print(np.mean(avgmax))
plt.title("9 periods")
plt.show()
avgmax=[]
viridis = mpl.colormaps['rainbow'].resampled(len(signallist[1]))
for i,el in enumerate(signallist[1]):
    elfreq = np.fft.fftfreq(len(el),0.05)[0:len(el)//2]
    elfft = 2*np.abs(np.fft.fft([x-np.mean(el) for x in el])[0:len(el)//2])/len(el)
    #plt.plot(np.linspace(0,len(el),len(el)),[x-np.mean(el) for x in el],c = viridis(i))
    plt.plot(elfreq,elfft)
    avgmax.append(np.max(elfft[10:]))
print(np.mean(avgmax))
plt.title("10 periods")
plt.show()
avgmax=[]
viridis = mpl.colormaps['rainbow'].resampled(len(signallist[1]))
for i,el in enumerate(signallist[2]):
    
    elfreq = np.fft.fftfreq(len(el),0.05)[0:len(el)//2]
    elfft = 2*np.abs(np.fft.fft([x-np.mean(el) for x in el])[0:len(el)//2])/len(el)
    #plt.plot(np.linspace(0,len(el),len(el)),[x-np.mean(el) for x in el],c = viridis(i))
    plt.plot(elfreq,elfft)
    avgmax.append(np.max(elfft[10:]))
print(np.mean(avgmax))
plt.title("11 periods")
plt.show()

    
#%%
print(len(Acimages2))
for i,sig in enumerate(Acimages2[-2:]):
    plt.plot(np.linspace(0,len(sig),len(sig)),[np.mean(x) for x in sig],label=i+8)
for i,sig in enumerate(Acimages[-2:]):
    plt.plot(np.linspace(0,len(sig),len(sig)),[np.mean(x) for x in sig],label=i+8)
for i,sig in enumerate(Acimages3[-2:]):
    plt.plot(np.linspace(0,len(sig),len(sig)),[np.mean(x) for x in sig],label=i+8)
plt.legend()
f
plt.show
#%%
from scipy.stats import linregress
for ac,dc in zip(imgs[:],DCs[5:]):
    plt.scatter(np.mean(dc),np.mean(ac))
    
    
meanac = [np.mean(ac) for ac in imgs]
meandc = [np.mean(dc) for dc in DCs]
slope, intercept, r_value, p_value, std_err = linregress(meandc, meanac)
print(r_value)
print(slope)
plt.xlabel("DC")
plt.ylabel("AC")
plt.show()

#%%
# for i,s in enumerate(signals):
#     plt.scatter(s[-1]-s[0],meanac[i])
def minmaxpos(lis):
    mn = np.min(lis)
    mx = np.max(lis)
    lis = [(el-mn)/(mx-mn) for el in lis]
    return lis
signaldiffs = [np.sum((s-np.mean(s))**2) for s in signals]
print(len(signaldiffs),len(meanac))
cmap = plt.cm.viridis.resampled(len(signals))

cellareas=[2710, 2808, 3088, 3077, 3284, 3683, 4211, 4348, 5496, 5634, 6835, 7940, 9276, 8971, 9925, 10425, 11390, 12857, 14117, 14262, 18705, 18922, 19088, 19249, 19259, 19440, 19653, 19810, 19479, 19811, 20313, 20625, 20801, 21069, 21106, 21524, 22032, 22117, 22741, 22994, 23859, 22917, 23053, 23252, 23488, 23653, 24045, 24441, 24799,24799]
cellareas = [x/(376*372) for x in cellareas]
# signaldiffs =minmaxpos(np.log(signaldiffs))
# cellareas = minmaxpos(cellareas)    
# meanac = minmaxpos(meanac)
# elev = 90
# azim = -180
# roll = 0
# ax.view_init(elev, azim, roll)
# ax.scatter(signaldiffs,meanac,cellareas,marker = 'o')

# ax.set_xlabel("Diff")
# ax.set_ylabel("meanac")
# ax.set_zlabel("areas")

# plt.show()

meanacscale = [x/y for x,y in zip(meanac,cellareas)]
plt.scatter(np.log2(signaldiffs),meanac,c = np.linspace(0,len(signals),len(signals)))
# plt.plot(np.linspace(0,len(DCs),len(DCs)),meanacscale)
# plt.show()
# plt.plot(np.linspace(0,len(DCs),len(DCs)),meandc)

    
#%%
for im in imgs:
    plt.imshow(im,vmin=0,vmax=65535)
    plt.show()
means = [np.mean(im) for im in imgs]
plt.plot(np.linspace(0,len(means),len(means)),means)
signalmean = [np.mean(s) for s in signals]
signalcorrmean = [np.mean(s) for s in signalcorrs]
plt.errorbar(np.linspace(0,len(signalmean),len(signalmean)),signalmean,fmt = '--')
plt.show()
plt.plot(np.linspace(0,len(signalmean),len(signalmean)),[mean-sigmean for mean,sigmean in zip(means,signalmean)])
plt.show()
#%%
plt.scatter(signalmean,means)
plt.show()
#%%
copy =  [deepcopy(i) for i in proms]
limit = 60
filteredresponses = deepcopy(copy[0])

filteredtail = np.zeros_like(copy[0])
filteredtail[imgs[0]>limit] = 1
print(np.unique(filteredtail))

filteredresponses[imgs[0] > limit] += 1
print(np.unique(filteredresponses))
fig,axs = plt.subplots(1,2)

axs[0].imshow(filteredresponses,cmap='rainbow',norm = 'linear',interpolation = None)


axs[1].imshow(filteredtail,cmap = 'rainbow',norm = 'linear',interpolation = None) 
plt.show()
#%%
imgcopy2 = deepcopy(imgs[0])
imgcopy2.flatten
resp = proms[0]
resp.flatten
h = imgcopy2[resp == 1]
plt.hist(h,bins = 100,density = True)

