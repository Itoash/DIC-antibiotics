from pyqtgraph.Qt import QtWidgets
from AnalysisGUI.utils.seg_utils import segmentDComni
import numpy as np 

class AnalysisBuffer:
    """
    Class for long-term storage of DC and AC images, as well as metadata.

    Stores AC/DC images, unique identifier names, and timestamps.
    Used for cell-by-cell analysis and displaying results.
    Optionally links to a SegmentationBuffer for synchronized sorting.
    """

    def __init__(self, segBuffer=None):
        """
        Initializes the AnalysisBuffer.

        Parameters
        ----------
        segBuffer : SegmentationBuffer, optional
            Reference to a SegmentationBuffer for synchronized sorting.
        """
        self.ACs = []
        self.DCs = []
        self.names = []
        self.times = []
        self.abstimes = []
        self.segBuffer = segBuffer

    def addImage(self, AC, DC, name, resort=True):
        """
        Adds AC and DC images with associated metadata.

        Parameters
        ----------
        AC : np.ndarray
            AC image loaded from AC Analysis tab.
        DC : np.ndarray
            DC image loaded from AC Analysis tab.
        name : str
            Directory name for origin of raw images.
        resort : bool, optional
            Whether to sort the buffer by absolute time after adding (default is True).

        Returns
        -------
        None
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
        """
        Sorts all stored data (and linked segmentation buffer, if present) by absolute time.

        Returns
        -------
        None
        """
        if self.segBuffer is not None:
            zipped = list(zip(self.abstimes, self.times, self.names, self.ACs, self.DCs, self.segBuffer.images, self.segBuffer.masks))
            zipped.sort(key=lambda x: x[0])
            self.abstimes, self.times, self.names, self.ACs, self.DCs, self.segBuffer.images, self.segBuffer.masks = map(list, zip(*zipped))

    def figureTime(self, name):
        """
        Extracts time information from the filename.

        Assumes naming convention:
        <magnification>_<method>_<voltage>_<ms>_<fps>_-<s/supr>_<date YYY/MM/DD>_<time 12H fmt> <AM/PM>

        Parameters
        ----------
        name : str
            Full filename of loaded directory.

        Returns
        -------
        time : tuple
            (hour, minutes) in 24h format.
        abstime : float
            Absolute time in minutes.
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
                hours = int(timestring[0]) + 12
                minutes = int(timestring[1:3])
                seconds = int(timestring[3:5])
            else:
                hours = int(timestring[0:2])
                minutes = int(timestring[2:4])
                seconds = int(timestring[4:6])

        time = (hours, minutes)
        abstime = 60 * hours + minutes + seconds / 60
        return time, abstime

    def clear(self):
        """
        Wipes all internal info clean.

        Returns
        -------
        None
        """
        self.ACs = []
        self.DCs = []
        self.names = []
        self.times = []
        self.abstimes = []


class SegmentationBuffer:
    """
    Buffer for storing DC and segmentation image data.

    Stores images and their corresponding segmentation masks.
    """

    def __init__(self):
        """
        Initializes the SegmentationBuffer.
        """
        self.images = []
        self.masks = []
        self.imagenet = None

    def addImage(self, img,params = None):
        """
        Adds a single image and computes its segmentation mask.

        Parameters
        ----------
        img : np.ndarray
            Image to add.

        Returns
        -------
        None
        """
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
        if params is not None:
            # If parameters are provided, use them for segmentation
            msk = segmentDComni([img],params=params)
        else:
            msk = segmentDComni([img])
        self.masks.append(msk[0])

    def addMultipleImages(self, imgs):
        """
        Adds multiple images and computes their segmentation masks.

        Parameters
        ----------
        imgs : list of np.ndarray
            List of images to add.

        Returns
        -------
        None
        """
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
        """
        Clears all stored images and masks.

        Returns
        -------
        None
        """
        self.images = []
        self.masks = []

    def remove(self, index):
        """
        Removes an image and its mask at the specified index.

        Parameters
        ----------
        index : int
            Index of the image/mask to remove.

        Returns
        -------
        None
        """
        self.images.pop(index)
        self.masks.pop(index)