from pyqtgraph.Qt import QtWidgets
from AnalysisGUI.utils.seg_utils import segmentDComni
import numpy as np 
class AnalysisBuffer:
    """
    Class for long-term storage of DC and AC images, as well as metadata;
    Contains AC/DC images, unique identifier names and (potentially non-unique) timestamps.
    Is used for final cell-by-cell analysis code, as well as displaying .
    """

    def __init__(self,segBuffer=None):
        self.ACs = []
        self.DCs = []
        self.names = []
        self.times = []
        self.abstimes = []
        self.segBuffer =segBuffer
    def addImage(self, AC, DC, name,resort = True):
        """
        Function for adding AC and DC images together, to prevent desync.
        Useful for cell-by-cell analysis, displaying results, and keeping track

        Parameters
        ----------
        AC : AC image loaded from AC Analysis tab
        DC : DC image loaded from AC Analysis tab
        name : Directory name for origin of raw images

        Returns
        -------
        None.

        """
        self.ACs.append(AC.astype(np.float64))
        self.DCs.append(DC.astype(np.float64))
        self.names.append(name)
        time, abstime = self.figureTime(name)
        self.times.append(time)
        self.abstimes.append(abstime)
        if resort:
            print(self.names)
            self.sortByAbsTime()
            print(self.names)
    def sortByAbsTime(self):
        if self.segBuffer is not None:
            zipped = list(zip(self.abstimes, self.times, self.names, self.ACs, self.DCs,self.segBuffer.images,self.segBuffer.masks))
            zipped.sort(key=lambda x: x[0])  # assuming you want to sort by self.abstimes
            self.abstimes, self.times, self.names, self.ACs, self.DCs,self.segBuffer.images,self.segBuffer.masks = map(list, zip(*zipped))
            
    def figureTime(self, name):
        """
        Function for figuring time from name, assuming it follows the naming convention:
        <magnification>_<method>_<voltage>_<ms>_<fps>_-<s/supr>_<date YYY/MM/DD>_<time 12H fmt>\ <AM/PM>
        Needs to get last separator for time
        Parameters
        ----------
        name : full filename of loaded dir
        Returns
        -------
        time: tuple of form (hour,minutes) in 24h format

        """
        timestring = name.split('_')[-1]
        ext = timestring.split(' ')[-1]
        timestring = timestring.split(' ')[0]
        if ext.upper() == 'AM':
            hours = int(timestring[0:2])
            minutes = int(timestring[2:4])
            seconds = int(timestring[4:6])
        elif ext.upper() == 'PM':
            if len(timestring) == 5:
                # highly unlikely to be 10 PM or after, fix if possible
                hours = int(timestring[0])+12
                minutes = int(timestring[1:3])
                seconds = int(timestring[3:5])
            else:
                # highly unlikely to be 10 PM or after, fix if possible
                hours = int(timestring[0:2])
                minutes = int(timestring[2:4])
                seconds = int(timestring[4:6])

        time = (hours, minutes)
        abstime = 60*hours+minutes+seconds/60
        return time, abstime

    def clear(self):
        """
        Wipes all internal info clean

        Returns
        -------
        None.

        """
        self.ACs = []
        self.DCs = []
        self.names = []
        self.times = []
        self.abstimes = []


# buffer for storing DC and segmentation image data
class SegmentationBuffer:
    def __init__(self):
        self.images = []
        self.masks = []
        self.imagenet = None

    def addImage(self, img):

        if len(self.images) != 0:
            shapelist = [im.shape for im in self.images]
            if img.shape != shapelist[0]:
                QtWidgets.QApplication.restoreOverrideCursor()
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText("Cannot add image!")
                msg.setInformativeText('Wrong dimensions')
                msg.setWindowTitle("TypeError")
                return

        self.images.append(img.astype(np.float64))

        msk = segmentDComni([img])
        self.masks.append(msk[0])

    def addMultipleImages(self, imgs):

        shapelist = set([im.shape for im in imgs])
        if len(shapelist) != 1:
            QtWidgets.QApplication.restoreOverrideCursor()
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Different sizes!")
            msg.setInformativeText(
                'Images have varying sizes. Select image folder with same size images throughout.')
            msg.setWindowTitle("TypeError")
            return
        if len(self.images) != 0:
            if shapelist[0] != self.images[0].shape:
                QtWidgets.QApplication.restoreOverrideCursor()
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText("Wrong sizes!")
                msg.setInformativeText(
                    'Images have different sizes than those already in buffer!')
                msg.setWindowTitle("TypeError")
                return

        msk = segmentDComni(imgs)
        self.masks += msk
        self.images += imgs

    def clear(self):
        self.images = []
        self.masks = []

    def remove(self, index):
        self.images.pop(index)
        self.masks.pop(index)