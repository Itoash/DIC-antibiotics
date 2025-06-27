import numpy as np
import sys
import time as tm
from AnalysisGUI.utils._ac_processor import  get_ac_data as get_AC_data

# holder for all analysis data in current file (need to add past repr as well)
class ImageHolder:
    def __init__(self, parent, raws,
                 frequency=1, framerate=16.7,
                 limits=None, signalmask=None):
        self.parent = parent  # for calls to update dashboard
        # set limits of analysis on startup
        if limits is None:
            limits = (0, len(raws)-1)
        # set default/received values
        self.raws = raws
        self.framerate = framerate
        self.frequency = frequency
        self.limits = limits
        self.filt_freqs = (0.1,6)
        self.interpolate = False
        self.filter = False
        self.codename = "Startup"
        # run update upon startup to generate images
        self.update(hardlimits=True)
    
    def setRaws(self, raws):
        self.raws = raws  # update raws
    def update(self, hardlimits=False):
        tic = tm.time()
        raws = np.moveaxis(self.raws.astype(np.float32),0,2)
        nperiods = self.raws.shape[0]/self.framerate
        print(f'Raw image shape is:{raws.shape}')
        print(f'Limits are {self.limits}')
        print(f'Nperiods is {nperiods}')
        if self.filt_freqs[1] > self.framerate/2:
            print(f'Warning: filter frequency {self.filt_freqs[1]} is higher than Nyquist frequency {self.framerate/2}. Setting to Nyquist frequency.')
            self.filt_freqs = (self.filt_freqs[0], self.framerate/2)
        if self.filt_freqs[0] < 0:
            print(f'Warning: filter frequency {self.filt_freqs[0]} is lower than 0. Setting to 0.')
            self.filt_freqs = (0, self.filt_freqs[1])
        if self.filt_freqs[0] >  self.frequency:
            print(f'Warning: filter frequency {self.filt_freqs[0]} is higher than frequency {self.frequency}. Setting to frequency.')
            self.filt_freqs = (self.frequency, self.filt_freqs[1])
        self.AC, self.DC, self.signaldata,self.limits = get_AC_data(raws,
                                                           frequency=self.frequency,
                                                           framerate=self.framerate,
                                                          start=self.limits[0],
                                                           end=self.limits[1],
                                                           hardlimits=hardlimits,interpolation=self.interpolate,filt = self.filter,periods =nperiods,filter_limits = self.filt_freqs)
        self.signaldata = (self.signaldata[0],np.moveaxis(self.signaldata[1],2,0))
        print(f'Ac shape:{self.AC.shape}')
        print(f'DC shape:{self.DC.shape}')
        print(f'Sig shape:{self.signaldata[0].shape}')
        print(f'Last time value:{self.signaldata[0][-1]}')
        print(f'Chosen dt:{self.signaldata[0][-1]-self.signaldata[0][-2]}')
        print(f'Time shape:{self.signaldata[1].shape}')
        toc = tm.time()-tic
        print(f"Processing took {toc:.3f} s")

    def reanalyze(self,frequency = None,
                  limits = None, interp = None,filt = None,hardlimits = False,filt_freqs = None):
        # reanalyze with new parameters
        if limits is not None:
            self.limits = limits
        if frequency is not None:
            self.frequency = frequency
        if interp is not None:
            self.interpolate = interp
        if filt is not None:  
            self.filter = filt
        if filt_freqs is not None:
            self.filt_freqs = filt_freqs
        print(f'new frequency {self.frequency}')
        print(f'new limits {self.limits}')
        print(f'new interpolation {self.interpolate}')
        print(f'new filter {self.filter}')
        print(f'framerate {self.framerate}')
        self.update(hardlimits=hardlimits)
        self.parent.updateAnalysis()
    def changeLimits(self, newlimits):
        self.limits = newlimits
        print(f'set new limits {self.limits}')
        self.update()
        self.parent.updateAnalysis()

    def changeFreq(self, newfreq):
        self.frequency = newfreq
        self.update()
        self.parent.updateAnalysis()
        # reset to some default values

    def reset(self):
        self.frequency = 1
        self.framerate = 16.7
        self.limits = (0, len(self.raws)-1)

# manager of analysis; has access to all panels;
# usually  requests to update pass through here