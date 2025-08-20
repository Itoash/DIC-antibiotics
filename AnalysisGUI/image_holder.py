import numpy as np
import sys
import time as tm
from AnalysisGUI.utils._ac_processor import get_ac_data as get_AC_data

# holder for all analysis data in current file (need to add past repr as well)
class ImageHolder:
    """
    A class to manage and process image data for analysis.

    Attributes:
    parent : object
        Reference to the parent object for dashboard updates.
    raws : np.ndarray
        Raw image data.
    frequency : float
        Frequency of the signal.
    framerate : float
        Frame rate of the image data.
    limits : tuple
        Limits for analysis (start and end indices).
    signalmask : np.ndarray, optional
        Mask for signal processing.
    filt_freqs : tuple
        Frequency range for filtering.
    interpolate : bool
        Whether to interpolate the data.
    filter : bool
        Whether to apply filtering.
    codename : str
        Identifier for the current analysis.
    AC : np.ndarray
        AC component of the processed data.
    DC : np.ndarray
        DC component of the processed data.
    signaldata : tuple
        Processed signal data.
    """

    def __init__(self, parent, raws, frequency=1, framerate=16.7, limits=None, signalmask=None):
        """
        Initialize the ImageHolder with raw image data and analysis parameters.

        Parameters:
        parent : object
            Reference to the parent object for dashboard updates.
        raws : np.ndarray
            Raw image data.
        frequency : float, optional
            Frequency of the signal (default is 1).
        framerate : float, optional
            Frame rate of the image data (default is 16.7).
        limits : tuple, optional
            Limits for analysis (start and end indices).
        signalmask : np.ndarray, optional
            Mask for signal processing.
        """
        self.parent = parent  # for calls to update dashboard
        # set limits of analysis on startup
        if limits is None:
            limits = (0, len(raws)-1)
        # set default/received values
        self.raws = raws
        self.framerate = framerate
        self.frequency = frequency
        self.limits = limits
        self.filt_freqs = (0.1, 6)
        self.interpolate = False
        self.filter = False
        self.codename = "Startup"
        # run update upon startup to generate images
        self.update(hardlimits=True)

    def setRaws(self, raws):
        """
        Update the raw image data.

        Parameters:
        raws : np.ndarray
            New raw image data.
        """
        self.raws = raws  # update raws

    def update(self, hardlimits=False):
        """
        Process the raw image data and update analysis results.

        Parameters:
        hardlimits : bool, optional
            Whether to enforce hard limits during processing (default is False).
        """
        tic = tm.time()
        raws = np.moveaxis(self.raws.astype(np.float32), 0, 2)
        nperiods = self.raws.shape[0] / self.framerate
        if self.filt_freqs[1] > self.framerate / 2:
            print(f'Warning: filter frequency {self.filt_freqs[1]} is higher than Nyquist frequency {self.framerate / 2}. Setting to Nyquist frequency.')
            self.filt_freqs = (self.filt_freqs[0], self.framerate / 2)
        if self.filt_freqs[0] < 0:
            print(f'Warning: filter frequency {self.filt_freqs[0]} is lower than 0. Setting to 0.')
            self.filt_freqs = (0, self.filt_freqs[1])
        if self.filt_freqs[0] > self.frequency:
            print(f'Warning: filter frequency {self.filt_freqs[0]} is higher than frequency {self.frequency}. Setting to selected frequency.')
            self.filt_freqs = (self.frequency, self.filt_freqs[1])
        self.AC, self.DC, self.signaldata, self.limits = get_AC_data(
            raws,
            frequency=self.frequency,
            framerate=self.framerate,
            start=self.limits[0],
            end=self.limits[1],
            hardlimits=hardlimits,
            interpolation=self.interpolate,
            filt=self.filter,
            periods=nperiods,
            filter_limits=self.filt_freqs
        )
        self.signaldata = (self.signaldata[0], np.moveaxis(self.signaldata[1], 2, 0))
        toc = tm.time() - tic
        print(f"Processing took {toc:.3f} s")

    def reanalyze(self, frequency=None, limits=None, interp=None, filt=None, hardlimits=False, filt_freqs=None):
        """
        Reanalyze the image data with updated parameters.

        Parameters:
        frequency : float, optional
            New frequency for analysis.
        limits : tuple, optional
            New limits for analysis.
        interp : bool, optional
            Whether to interpolate the data.
        filt : bool, optional
            Whether to apply filtering.
        hardlimits : bool, optional
            Whether to enforce hard limits during processing.
        filt_freqs : tuple, optional
            New frequency range for filtering.
        """
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
        self.update(hardlimits=hardlimits)
        self.parent.updateAnalysis()

    def changeLimits(self, newlimits):
        """
        Update the limits for analysis.

        Parameters:
        newlimits : tuple
            New limits for analysis.
        """
        self.limits = newlimits
        print(f'Set new limits {self.limits}')
        self.update()
        self.parent.updateAnalysis()

    def changeFreq(self, newfreq):
        """
        Update the frequency for analysis.

        Parameters:
        newfreq : float
            New frequency for analysis.
        """
        self.frequency = newfreq
        self.update()
        self.parent.updateAnalysis()

    def reset(self):
        """
        Reset the analysis parameters to default values.
        """
        self.frequency = 1
        self.framerate = 16.7
        self.limits = (0, len(self.raws) - 1)