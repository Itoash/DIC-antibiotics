import numpy as np
import sys
import time as tm
if sys.platform == 'darwin':
    from AnalysisGUI.utils.ac_utils import get_AC_data
else:
    from AnalysisGUI.utils.ac_utils_cuda import get_AC_data

# holder for all analysis data in current file (need to add past repr as well)
class ImageHolder:
    def __init__(self, parent, raws,
                 frequency=1, framerate=16.7,
                 limits=None, signalmask=None):
        self.parent = parent  # for calls to update dashboard
        # set limits of analysis on startup
        if limits is None:
            limits = (100, len(raws)-1)
        # set default/received values
        self.raws = raws
        self.framerate = framerate
        self.frequency = frequency
        self.limits = limits
        self.interpolate = True
        self.filter = True
        self.codename = "Startup"
        # run update upon startup to generate images
        self.update(hardlimits=True)

    def setRaws(self, raws):
        self.raws = raws  # update raws

    def update(self, hardlimits=False):
        tic = tm.time()
        self.AC, self.DC, self.signaldata, _ = get_AC_data(self.raws.astype(np.float32).copy(),
                                                           frequency=self.frequency,
                                                           framerate=self.framerate,
                                                           start=self.limits[0],
                                                           end=self.limits[1],
                                                           hardlimits=hardlimits,interpolation=self.interpolate,filt = self.filter)
        toc = tm.time()-tic
        print(f"Processing took {toc:.3f} s")

    def reanalyze(self,frequency = None,
                  limits = None, interp = None,filt = None,hardlimits = False):
        # reanalyze with new parameters
        if limits is not None:
            self.limits = limits
        if frequency is not None:
            self.frequency = frequency
        if interp is not None:
            self.interpolate = interp
        if filt is not None:  
            self.filter = filt
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