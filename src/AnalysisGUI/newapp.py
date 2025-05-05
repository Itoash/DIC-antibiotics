#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 08:03:22 2025

@author: victorionescu
App scripts
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets, QtCore
from copy import deepcopy
from AnalysisGUI.cell_viewer import CellViewer
import os
import cv2
import platform
from AnalysisGUI.utils.TrackerModule import obtain_network, stabilize_images
from AnalysisGUI.utils.seg_utils import segmentDComni
from AnalysisGUI.utils.ac_utils import get_AC_data
import time as tm
from AnalysisGUI.timeseries_viewer import DataVisualizerApp
from AnalysisGUI.utils.cellprocessor import process_cells
from copy import deepcopy

# from track_utils import obtain_network #maybe later
pg.setConfigOption('imageAxisOrder', 'row-major')

# function to artificially add channels to images/list of images,
# as they are monochrome 16-bit


def multichan(imagedata):
    chan1 = np.copy(imagedata)
    chan2 = np.copy(imagedata)
    chan3 = np.copy(imagedata)
    multi = np.stack([chan1, chan2, chan3], axis=-1)
    return multi
# holds all images


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
        self.codename = "Startup"
        # run update upon startup to generate images
        self.update(hardlimits=True)

    def setRaws(self, raws):
        self.raws = raws  # update raws

    def update(self, hardlimits=False):
        tic = tm.time()
        self.AC, self.DC, self.signaldata, _ = get_AC_data(self.raws.astype(np.float64),
                                                           frequency=self.frequency,
                                                           framerate=self.framerate,
                                                           start=self.limits[0],
                                                           end=self.limits[1],
                                                           hardlimits=hardlimits)
        toc = tm.time()-tic
        print(f"Processing took {toc}:.3f s")

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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # make image holder with random data right away;
        # this calls an analysis on startup but it's fine
        self.imageData = ImageHolder(self,
                                     np.asarray([np.random.uniform(0, 1, (300, 300))
                                                 for i in range(400)]))
        # make buffer for seg images
        self.segBuffer = SegmentationBuffer()
        self.analysisImages = AnalysisBuffer(self.segBuffer)
        # make a dock area that auto-inits the predetermined plots
        self.docks = AnalysisArea(self)
        self.setCentralWidget(self.docks)

        # update all plots, create menus and set a title
        self.updateAnalysis()
        self.createActions()
        self.createMenu()
        self.setWindowTitle('Analiza AC')

    def updateAnalysis(self):
        # if there is data, post it to all panels;
        # all panels have acces to imageHolder and pull
        # from there on update
        if self.imageData:
            self.docks.wDIC.update()
            self.docks.wDC.update()
            self.docks.wAC.update()
            # updateSignals is unique in that it also includes
            # a masking function for ROI interaction;
            # the rest just display whatever is in their slot
            # in the ImageHolder
            self.docks.wSig.updateSignals()

    def resetAnalysis(self):
        # Reset image data, and then update plots
        self.imageData.reset()
        self.updateAnalysis()

    def open(self):
        # to fix, but obsolete now we have the file tree
        fileName = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                              "Open File",
                                                              QtCore.QDir.currentPath())
        # fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '','Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)

        if fileName:
            self.loadstack(fileName)

    def loadstack(self, filename=None):
        # get elected directory and run through some checks:
        if filename is None or filename is False:
            index = self.docks.treeview.currentIndex()
            filename = self.docks.treeview.model.filePath(index)
        # Check 1: if it's not a directory, display message and abort
        print(filename)
        if not os.path.isdir(filename):
            QtWidgets.QApplication.restoreOverrideCursor()
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Not a directory!")
            msg.setInformativeText(
                'Please select a directory containing image data of the same dimensions!')
            msg.setWindowTitle("TypeError")
            msg.exec_()
            return

        # Check 1 passed: scan names to see if there are valid file formats one level down
        files = [f for f in os.listdir(filename) if f != '.DS_Store']
        validformats = ['png', 'tif', 'jpg']
        formatcounts = [0, 0, 0]
        validfiles = [f for f in files if f.split('.')[-1] in validformats]
        for f in validfiles:
            ext = f.split('.')[-1]
            idx = validformats.index(ext)
            formatcounts[idx] += 1
        singleformat = validformats[np.argmax(formatcounts)]
        validfiles = [f for f in validfiles if f.split(
            '.')[-1] == singleformat]

        # Check 2: see if there are any valid files; if not, display message and abort
        if len(validfiles) == 0:

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No valid files!")
            msg.setInformativeText(
                'Chosen directory contains no valid image files (*.tif,*.png,*.jpg)')
            msg.setWindowTitle("TypeError")
            msg.exec_()

            return
        # if there are, sort files and load images, put their sizes in set
        validfiles = list(sorted(validfiles, key=lambda x: int(x[-8:-4])))
        images = [cv2.imread(os.path.join(filename, name), -1)
                  for name in validfiles]

        image_sizes = set([img.shape for img in images])

        # Check 3: if image sizes are diffferent, print message and exit
        if len(image_sizes) != 1:

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Images have different sizes!")
            msg.setInformativeText(
                'Chosen directory contains images of different sizes.)')
            msg.setWindowTitle("SizeError")
            msg.exec_()
            return

        # get fps info from filename (CHANGE IF NAMING CONVENTION CHANGES)
        params = filename.split('_')
        fps = [el for el in params if 'fps' in el][0]
        fps = fps[:-3]
        fps = fps.split('v')
        try:
            framerate = int(fps[0])+int(fps[1])/10
        except IndexError:
            framerate = int(fps[0])
        # limits are endpoints
        limits = (100, len(images)-1)
        images = np.asarray(images)
        _,codename = os.path.split(filename)
        print(codename)
        self.imageData.codename = codename
        self.imageData.setRaws(images)  # change raw image data
        self.imageData.limits = limits  # change limits in imageData object
        self.imageData.framerate = framerate  # change frate in imageData object
        self.imageData.update()  # update images internally (run AC)
        self.updateAnalysis()  # update Plots

    def segment(self):
        if len(self.segBuffer.images) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No images in buffer!")
            msg.setInformativeText('Segmentation buffer contains no images')
            msg.setWindowTitle("IndexError")
            msg.exec_()
            return
        self.segmentor = SegmentWindow(self.segBuffer,self.analysisImages)
        self.segmentor.show()

    def track(self):
        if len(self.segBuffer.images) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No images in buffer!")
            msg.setInformativeText('Segmentation buffer contains no images')
            msg.setWindowTitle("IndexError")
            msg.exec_()
            return
        self.tracker = TrackWindow(self.segBuffer, self.analysisImages)
        self.tracker.show()

    def addImages(self):
        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open File", QtCore.QDir.currentPath())
        if not os.path.isdir(filename):
            QtWidgets.QApplication.restoreOverrideCursor()
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Not a directory!")
            msg.setInformativeText(
                'Please select a directory containing image data of the same dimensions!')
            msg.setWindowTitle("TypeError")
            msg.exec_()
            return

        # Check 1 passed: scan names to see if there are valid file formats one level down
        files = [f for f in os.listdir(filename) if f != '.DS_Store']
        validformats = ['png', 'tif', 'jpg']

        validfiles = [f for f in files if f.split('.')[-1] in validformats]

        # Check 2: see if there are any valid files; if not, display message and abort
        if len(validfiles) == 0:

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No valid files!")
            msg.setInformativeText(
                'Chosen directory contains no valid image files (*.tif,*.png,*.jpg)')
            msg.setWindowTitle("TypeError")
            msg.exec_()

            return
        # if there are, sort files and load images, put their sizes in set
        validfiles = list(sorted(validfiles))
        images = [cv2.imread(os.path.join(filename, name), -1)
                  for name in validfiles]
        image_sizes = set([img.shape for img in images])

        # Check 3: if image sizes are diffferent, print message and exit
        if len(image_sizes) != 1:

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Images have different sizes!")
            msg.setInformativeText(
                'Chosen directory contains images of different sizes.)')
            msg.setWindowTitle("SizeError")
            msg.exec_()
            return
        self.segBuffer.addMultipleImages(images)

    def loadDay(self):
        index = self.docks.treeview.currentIndex()
        filename = self.docks.treeview.model.filePath(index)
        if not os.path.isdir(filename):
            QtWidgets.QApplication.restoreOverrideCursor()
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Not a directory!")
            msg.setInformativeText(
                'Please select a directory containing image data of the same dimensions!')
            msg.setWindowTitle("TypeError")
            msg.exec_()
            return
        files = [os.path.join(filename, f) for f in os.listdir(
            filename) if '.DS_Store' not in f and os.path.isdir(os.path.join(filename, f)) and '_' in f]
        timestrings = [f.split("_")[-1] for f in files]
        ext = [f.split(' ')[-1] for f in timestrings]
        timestrings = [f.split(' ')[0] for f in timestrings]
        times = []

        for ts, ex in zip(timestrings, ext):
            if ex.upper() == 'AM':
                times.append(60*int(ts[0:2])+int(ts[2:4]))
            else:
                if len(ts) == 5:
                    times.append(60*int(ts[0:1])+int(ts[1:3])+12*60)
                else:
                    times.append(int(ts[1:3])+12*60)
        times = [t-times[0] for t in times]
        files = [f for _, f in sorted(zip(times, files))]
        self.clearBuffer()
        tic = tm.time()
        for i, f in enumerate(files):
            if os.path.isdir(f):
                self.loadstack(f)
                self.addImage(resort=False)
            else:
                # something happened to directory during segmentation
                break
        toc = tm.time()-tic
        print(f"Segmentation and processing took {toc}:3fs")

    def addImage(self, img=None,resort = True):

        DC = self.imageData.DC
        AC = self.imageData.AC
        name = self.imageData.codename
        self.segBuffer.addImage(DC)
        self.analysisImages.addImage(AC, DC, name,resort=resort)
        print('Added image!')
        print(
            f'Segbuffer: masks/DCs{len(self.segBuffer.masks),len(self.segBuffer.images)}')
        print(
            f'Analysis buff: ACs/DCs{len(self.analysisImages.ACs),len(self.analysisImages.DCs)}')

    def clearBuffer(self):
        self.segBuffer.clear()
        self.analysisImages.clear()

    def about(self):
        # whatever this is
        QtWidgets.QMessageBox.about(self, "About Image Viewer",
                                    "<p>The <b>DIC Analyzer</b> works fine don't worry about it...</p>")

    def createActions(self):
        # make actions and shortcuts for important top-level commands
        self.openAct = QtWidgets.QAction(
            "&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.loadAct = QtWidgets.QAction(
            "&Load Stack", self, shortcut="Ctrl+L", triggered=self.loadstack)
        self.loadDayAct = QtWidgets.QAction(
            "&Load Folder of Stacks", self, shortcut="Ctrl+A", triggered=self.loadDay)
        self.exitAct = QtWidgets.QAction(
            "&Exit", self, shortcut="Ctrl+Q", triggered=self.closeApp)
        self.segmentAct = QtWidgets.QAction(
            "&segment DC", self, shortcut="Ctrl+F", triggered=self.segment)
        self.aboutAct = QtWidgets.QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QtWidgets.QAction(
            "About &Qt", self, triggered=QtWidgets.qApp.aboutQt)
        self.addimageAct = QtWidgets.QAction(
            "Add image to seg. buffer", self, shortcut="Ctrl+T", triggered=self.addImage)
        self.startsegAct = QtWidgets.QAction(
            "Start segmentation", self, shortcut="Ctrl+F", triggered=self.segment)
        self.clearsegAct = QtWidgets.QAction(
            "Clear seg. buffer", self, shortcut="Ctrl+D", triggered=self.clearBuffer)
        self.importSeg = QtWidgets.QAction(
            "Import files into seg. buffer", self, shortcut="Ctrl+I", triggered=self.addImages)
        self.trackAct = QtWidgets.QAction(
            "Start tracking with seg. buffer", self, shortcut="Ctrl+K", triggered=self.track)

    def closeApp(self):
        if 'segmentor' in self.__dict__.keys():
            self.segmentor.close()
        if 'tracker' in self.__dict__.keys():
            self.tracker.close()
        self.close()

    def createMenu(self):
        # display commands in menubar
        self.fileMenu = QtWidgets.QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addAction(self.loadDayAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        self.helpMenu = QtWidgets.QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)
        self.segMenu = QtWidgets.QMenu("&Segmentation", self)
        self.segMenu.addAction(self.addimageAct)
        self.segMenu.addAction(self.startsegAct)
        self.segMenu.addAction(self.clearsegAct)
        self.segMenu.addAction(self.importSeg)
        self.trackMenu = QtWidgets.QMenu("&Tracking", self)
        self.trackMenu.addAction(self.trackAct)
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.segMenu)
        self.menuBar().addMenu(self.trackMenu)
        self.menuBar().addMenu(self.helpMenu)
        
    # this is herer to limit access between plots; called by images when ROI is open
    # to update signal plots
    def updateSignals(self, mask):
        self.docks.wSig.updateSignals(mask)


# holds docks; can be removed and incorporated into mainwindow, but it's fine for now
class AnalysisArea(DockArea):
    # Do not change this init; the layout is very finnicky and took a while to setup
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.fileExplorerDock = Dock("File Explorer", size=(100, 100))
        self.ACDock = Dock("AC", size=(500, 300))
        self.signalsDock = Dock("Signal plot", size=(500, 200))
        self.DICDock = Dock("Raw images", size=(500, 500))
        self.FFTDock = Dock("FFT", size=(500, 200))
        self.DCDock = Dock("DC images", size=(500, 500))
        self.addDock(self.fileExplorerDock, 'left')
        self.addDock(self.signalsDock, 'right')
        self.addDock(self.ACDock, 'right', self.fileExplorerDock)
        self.addDock(self.DICDock, 'right')
        self.addDock(self.DCDock, 'top', self.signalsDock)
        self.moveDock(self.DICDock, 'top', self.signalsDock)
        self.moveDock(self.DCDock, 'above', self.DICDock)
        self.moveDock(self.fileExplorerDock, 'left', self.ACDock)
        self.treeview = FileExplorer(self)
        self.fileExplorerDock.addWidget(self.treeview)
        self.wDIC = DICWidget(self.parent, self.parent.imageData)
        self.wDC = DCWidget(self.parent, self.parent.imageData)
        self.wAC = ACWidget(self.parent, self.parent.imageData)
        self.DICDock.addWidget(self.wDIC)
        self.DCDock.addWidget(self.wDC)
        self.ACDock.addWidget(self.wAC)
        self.wSig = Signals(self.parent, self.parent.imageData)
        self.signalsDock.addWidget(self.wSig)


# holds the file system and a useless context menu;
# only good part is resizes
# CHANGE PRESET ROOT PATH AND EXPANSIONS WHEN PORTING
class FileExplorer(QtWidgets.QTreeView):
    def __init__(self, parent):
        super().__init__(parent)
        self.header().setSectionResizeMode(3)
        self.model = QtWidgets.QFileSystemModel()
        self.model.setRootPath(QtCore.QDir.homePath())
        self.setModel(self.model)
        self.setExpanded(self.model.index(QtCore.QDir.homePath()), True)
        self.resizeColumnToContents(0)
        self.setExpanded(self.model.index(os.path.join(QtCore.QDir.homePath(),'Users')), True)
        self.resizeColumnToContents(0)

# controls viewing of raw images


class DICWidget(pg.ImageView):
    def __init__(self, parent, imagesource):
        super().__init__(discreteTimeLine=True, levelMode='mono')
        self.parent = parent
        self.imageSource = imagesource  # link to ImageHolder to update analysis
        self.setImage(multichan(self.imageSource.raws), xvals=np.linspace(
            0, self.imageSource.raws.shape[0]/self.imageSource.framerate, self.imageSource.raws.shape[0]))
        # hold internal ref to DICarray (idk why, you always call the imageHolder on updates to check for changes)
        self.DICarray = self.imageSource.raws

        # store cursor positions in scene and global
        self.cursor_position = QtCore.QPoint(0, 0)
        self.global_pos = QtCore.QPoint(0, 0)

        # connect mouseMove and time change signals to display tooltip with px info
        self.imageItem.scene().sigMouseMoved.connect(self.mouseMove)
        self.sigTimeChanged.connect(self.tChange)

    def tChange(self, ind, times):
        # get array indexes for mouse
        row, col = int(self.cursor_position.x()), int(self.cursor_position.y())

        # if not in array do nothing
        if not ((0 <= row < self.DICarray.shape[1]) and (0 <= col < self.DICarray.shape[2])):
            return
        # else display tooltip; can change tobe only on arrows to reduce bugs, but works for now
        message = f'X = {self.cursor_position.x():.1f};\nY = {self.cursor_position.y():.1f};\n' + \
            f'Value = {self.image[ind,row,col,0]};'
        QtWidgets.QToolTip.showText(self.global_pos, message)

    def mouseMove(self, abs_pos):
        # same as above, but also stor mouse pos in local variable
        pos = self.imageItem.mapFromScene(abs_pos)
        row, col = int(pos.y()), int(pos.x())

        if (0 <= row < self.DICarray.shape[1]) and (0 <= col < self.DICarray.shape[2]):
            message = f'X = {row};\nY = {col};\n' + \
                f'Value = {self.DICarray[self.currentIndex,row,col,0]};'
            self.setToolTip(message)

        else:
            self.setToolTip('')
        global_pos = QtCore.QPoint(int(abs_pos.x()), int(abs_pos.y()))
        self.global_pos = self.mapToGlobal(global_pos)
        self.cursor_position = pos

    def update(self):
        # update based on limits in imageHolder
        lims = self.imageSource.limits
        print(f'updated Raws with limits {lims}')
        DICdata = self.imageSource.raws[lims[0]:lims[1]]
        print(f'New no of images = {len(DICdata)}')
        DICdata = multichan(DICdata)
        # recompute t axis to hold time in seconds in absolute terms
        self.setImage(DICdata, xvals=np.linspace(
            lims[0]/self.imageSource.framerate, lims[1]/self.imageSource.framerate, int(lims[1]-lims[0])))
        self.DICarray = DICdata
        print('Set in DIC')

    def roiChanged(self):
        # define super method
        super().roiChanged()
        if self.image is None:
            return
        # gets bounding box of roi, and gets coords
        image = self.getProcessedImage()
        colmaj = self.imageItem.axisOrder == 'col-major'
        if colmaj:
            axes = (self.axes['x'], self.axes['y'])
        else:
            axes = (self.axes['y'], self.axes['x'])

        maskslice, _ = self.roi.getArraySlice(
            image.view(np.ndarray), img=self.imageItem, axes=axes)
        mask = np.zeros_like(self.imageSource.AC)

        # do not trust slices
        xstart = maskslice[0].start
        xstop = maskslice[0].stop
        ystart = maskslice[1].start
        ystop = maskslice[1].stop
        mask[xstart:xstop, ystart:ystop] = 1

        self.parent.updateSignals(mask)  # proxy call to signals plot


# holds signal plots, and has functionality for updating analysis range (by proxy)
class Signals(QtWidgets.QWidget):  # class for handling signal data and updating plots
    def __init__(self, parent, imagesource):
        super().__init__()
        # include references to mainwindow and imagesource
        self.parent = parent
        self.imageSource = imagesource

        # set up a toolbar for updating/resetting analysis
        self.toolbar = QtWidgets.QToolBar()
        self.updateButton = QtWidgets.QAction("Update Analysis..", self)
        self.updateButton.setStatusTip("Redo limits and update analysis")
        self.updateButton.triggered.connect(self.updateLimits)
        self.toolbar.addAction(self.updateButton)
        self.resetButton = QtWidgets.QAction("Reset Analysis", self)
        self.resetButton.setStatusTip("Reset limits to 0,end")
        self.resetButton.triggered.connect(self.resetLimits)
        self.toolbar.addAction(self.resetButton)

        # sets up plots
        self.signalPlot = pg.PlotWidget(title='Averaged signal', labels={
                                        'bottom': 'T[s]', 'left': 'I'})
        self.FFTPlot = pg.PlotWidget(title='FFT of averaged signal', labels={
                                     'bottom': 'f[Hz]', 'left': 'A'})

        # gets data from imagesource
        self.sig = np.mean(self.imageSource.signaldata[1], axis=(1, 2))
        self.time = self.imageSource.signaldata[0]
        self.dt = 1/self.imageSource.framerate
        self.fft = np.fft.fft(self.sig)
        self.freq = np.fft.fftfreq(len(self.fft), self.dt)[0:len(self.fft)//2]
        self.fft = 2*np.abs(self.fft[0:len(self.fft)//2])/len(self.fft)

        # plot data; need plotDataItem reference to clip region to
        self.sigdata = self.signalPlot.plot(np.array([self.time, self.sig]).T)
        self.FFTPlot.plot(np.array([self.freq, self.fft]).T)

        # make region: to endpoints of signal,bounded to endpoints of signal, track endpoints
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.signalPlot.addItem(self.region, ignoreBounds=True)
        self.region.setClipItem(self.sigdata)
        self.region.sigRegionChanged.connect(self.updateRegions)
        self.minX = 0
        self.maxX = self.time[-1]
        self.region.setRegion((0, self.time[-1]))
        self.region.setBounds((0, self.time[-1]))

        # make a layout
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setMenuBar(self.toolbar)
        self.setLayout(self.layout)
        self.layout.addWidget(self.signalPlot)
        self.layout.addWidget(self.FFTPlot)
        self.show()

    def resetLimits(self):
        # reset limits from imagesource raws: those never change, and we can get back to start
        limits = (0, int(len(self.imageSource.raws)))
        self.imageSource.changeLimits(limits)

    def updateLimits(self):
        # set limits to those enclosed by the region; minX/maxX are updated on region move
        self.imageSource.changeLimits(
            (int(self.minX/self.dt), int(self.maxX/self.dt)))

    def updateRegions(self):
        # here we update region limits in case we want to redo analysis
        self.minX, self.maxX = self.region.getRegion()

    def getFFT(self):
        # small method for quickly geting fft without having to write this everytime
        self.fft = np.fft.fft(self.sig)
        self.freq = np.fft.fftfreq(len(self.fft), self.dt)[0:len(self.fft)//2]
        self.fft = 2*np.abs(self.fft[0:len(self.fft)//2])/len(self.fft)

    def updateSignals(self, mask=None):
        # check if mask
        if mask is None:
            mask = np.ones_like(self.imageSource.AC)

        # only include parts where signal is not masked by ROI
        self.sig = self.imageSource.signaldata[1][:, mask == 1].mean(axis=(1))

        # clear plots and replot
        self.time = self.imageSource.signaldata[0]
        self.getFFT()
        self.signalPlot.getPlotItem().clear()
        self.signalPlot.plot(np.array([self.time, self.sig]).T)
        self.FFTPlot.getPlotItem().clear()
        self.FFTPlot.plot(np.array([self.freq, self.fft]).T)

        # reset region, identical to init
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.signalPlot.addItem(self.region, ignoreBounds=True)
        self.region.setClipItem(self.sigdata)
        self.region.sigRegionChanged.connect(self.updateRegions)
        self.minX = 0
        self.maxX = self.time[-1]
        self.region.setRegion([self.minX, self.maxX])
        self.region.setBounds((0, self.time[-1]))


# fairly standard extension of ImageView, includes tooltip px values and links to ROI-to-signal
class DCWidget(pg.ImageView):
    def __init__(self, parent, imagesource):
        super().__init__(levelMode='mono')

        self.parent = parent
        self.imageSource = imagesource
        chans = self.imageSource.DC
        default = multichan(chans)
        self.setImage(default)
        # kern = np.array([
        #     [1]
        # ])
        # self.getImageItem().setDrawKernel(kern, mask=kern, center=(1,1), mode='add')

        self.DCarray = chans
        self.cursor_position = QtCore.QPoint(0, 0)
        self.getImageItem().scene().sigMouseMoved.connect(self.mouseMove)

    def mouseMove(self, abs_pos):

        pos = self.imageItem.mapFromScene(abs_pos)
        row, col = int(pos.y()), int(pos.x())

        if (0 <= row < self.DCarray.shape[0]) and (0 <= col < self.DCarray.shape[1]):
            message = f'X = {pos.x():.1f};\nY = {pos.y():.1f};\n' + \
                f'Value = {self.DCarray[row,col]:.3f};'
            self.setToolTip(message)
            global_pos = QtCore.QPoint(int(abs_pos.x()), int(abs_pos.y()))
            self.global_pos = self.mapToGlobal(global_pos)
            self.cursor_position = pos
        else:
            self.setToolTip('')

    def update(self):
        chans = self.imageSource.DC
        self.DCarray = chans
        chans = multichan(chans)
        self.setImage(chans)
        print('Set in DC')

    def roiChanged(self):
        super().roiChanged()
        if self.image is None:
            return

        image = self.getProcessedImage()

        # getArrayRegion axes should be (x, y) of data array for col-major,
        # (y, x) for row-major
        # can't just transpose input because ROI is axisOrder aware
        colmaj = self.imageItem.axisOrder == 'col-major'
        if colmaj:
            axes = (self.axes['x'], self.axes['y'])
        else:
            axes = (self.axes['y'], self.axes['x'])

        maskslice, _ = self.roi.getArraySlice(
            image.view(np.ndarray), img=self.imageItem, axes=axes)
        mask = np.zeros_like(self.imageSource.AC)

        xstart = maskslice[0].start
        xstop = maskslice[0].stop
        ystart = maskslice[1].start
        ystop = maskslice[1].stop
        mask[xstart:xstop, ystart:ystop] = 1

        self.parent.updateSignals(mask)


# very similar to DC; can collapse these two
class ACWidget(pg.ImageView):
    def __init__(self, parent, imagesource):
        super().__init__(levelMode='mono')
        self.parent = parent
        self.imageSource = imagesource
        chans = self.imageSource.AC
        default = multichan(chans)
        self.setImage(default)

        self.ACarray = chans
        self.cursor_position = QtCore.QPoint(0, 0)
        self.imageItem.scene().sigMouseMoved.connect(self.mouseMove)

    def mouseMove(self, abs_pos):

        pos = self.imageItem.mapFromScene(abs_pos)
        row, col = int(pos.y()), int(pos.x())

        if (0 <= row < self.ACarray.shape[0]) and (0 <= col < self.ACarray.shape[1]):
            message = f'X = {pos.x():.1f};\nY = {pos.y():.1f};\n' + \
                f'Value = {self.ACarray[row,col]:.3f};'
            self.setToolTip(message)
            global_pos = QtCore.QPoint(int(abs_pos.x()), int(abs_pos.y()))
            self.global_pos = self.mapToGlobal(global_pos)
            self.cursor_position = pos
        else:
            self.setToolTip('')

    def updateACdata(self, ACdata):
        self.ACarray = ACdata
        ACdata = multichan(ACdata)
        self.setImage(ACdata)
        print('Set in AC')

    def roiChanged(self):
        super().roiChanged()
        if self.image is None:
            return

        image = self.getProcessedImage()

        colmaj = self.imageItem.axisOrder == 'col-major'
        if colmaj:
            axes = (self.axes['x'], self.axes['y'])
        else:
            axes = (self.axes['y'], self.axes['x'])

        maskslice, _ = self.roi.getArraySlice(
            image.view(np.ndarray), img=self.imageItem, axes=axes)
        mask = np.zeros_like(self.imageSource.AC)

        xstart = maskslice[0].start
        xstop = maskslice[0].stop
        ystart = maskslice[1].start
        ystop = maskslice[1].stop
        mask[xstart:xstop, ystart:ystop] = 1

        self.parent.updateSignals(mask)

    def update(self):
        chans = self.imageSource.AC
        self.ACarray = chans
        chans = multichan(chans)
        self.setImage(chans)
        print('Set in AC')


# window holder for DC/Seg data: TBD
# logic to:
    # auto segment first image
    # display DC and segmentation results in side-by-side
        # no docks this time, static widgets
        # change LUT of segmentation results
        # also add pixel values for segmentation results
    # add menu for:
        # overlayiing results on DC image and updating in real time (slider that's always on maybe?)
        # overlay/outline with plotcurveitem maybe?
        # draw mode: color picker size picker etc.
        # have to join oversegmented masks
        # have to separate undersegmented masks
        # clean up masks with tracking/convexity/size criteria (later)
        # add image to segmentation (from here or main window)
    # handle adding and saving images in a buffer (DC + seg image)
        # can re-open seg image from buffer if closed
        # can save seg results
        # implement saving all images (AC, DC) in MainWindow
    # segmentation correction routines?
    # also add tracking tree:
        # should highlight cells when clicked:
        # can link/unlink/delete items
        # break merges
        # fix over/under segmentation (with ellipses for now)
class SegmentWindow(QtWidgets.QMainWindow):
    def __init__(self, segBuffer,analysisBuffer):  # always have to init with an image
        super().__init__()
        self.segBuffer = segBuffer
        self.analysisBuffer = analysisBuffer
        self.wid = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QHBoxLayout()
        self.brushsize = 3
        self.brushcolor = 0
        # set up a toolbar for updating/resetting analysis
        self.setToolbar()

        self.DCplot = OverlayImage(self)
        self.DCplot.sigChangedDefaultCol.connect(self.changeCol)
        self.Segplot = pg.ImageView()
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})

        img = np.asarray(self.segBuffer.masks)

        self.DCplot.setOverlay(img)
        self.Segplot.setImage(img, axes={'t': 0, 'x': 2, 'y': 1, 'c': None})
        self.Segplot.setColorMap(pg.colormap.get('viridis'))

        # set up links, clear unneeded UI
        self.DCplot.view.setXLink(self.Segplot.view)
        self.DCplot.view.setYLink(self.Segplot.view)
        self.Segplot.view.setXLink(self.DCplot.view)
        self.Segplot.view.setXLink(self.DCplot.view)
        self.DCplot.sigTimeChanged.connect(self.Segplot.setCurrentIndex)
        self.Segplot.sigTimeChanged.connect(self.DCplot.setCurrentIndex)
        self.DCplot.ui.roiBtn.hide()
        self.DCplot.ui.menuBtn.hide()
        self.Segplot.ui.roiBtn.hide()
        self.Segplot.ui.menuBtn.hide()
        self.DCplot.view.setMenuEnabled(False)
        self.Segplot.view.setMenuEnabled(False)

        self.layout.addWidget(self.DCplot)
        self.layout.addWidget(self.Segplot)
        self.layout.setMenuBar(self.toolbar)
        self.wid.setLayout(self.layout)
        self.setCentralWidget(self.wid)

        self.showMaximized()
        self.DCplot.autoRange()
        self.Segplot.autoRange()
        
    def deleteImage(self):
        currentIdx = self.DCplot.currentIndex
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            self.segBuffer.images.pop(currentIdx)
            self.segBuffer.masks.pop(currentIdx)
            self.analysisBuffer.ACs.pop(currentIdx)
            self.analysisBuffer.DCs.pop(currentIdx)
            self.analysisBuffer.names.pop(currentIdx)
            self.analysisBuffer.times.pop(currentIdx)
            self.analysisBuffer.abstimes.pop(currentIdx)
        self.resetImages()
            
    def resetImages(self):
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        img = np.asarray(self.segBuffer.masks)
        self.DCplot.setOverlay(img)
        self.Segplot.setImage(img, axes={'t': 0, 'x': 2, 'y': 1, 'c': None})
        self.Segplot.setColorMap(pg.colormap.get('viridis'))

    def checkpoint(self):
        imgs = self.DCplot.overlay
        imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        self.segBuffer.masks = imgs
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        img = np.asarray(self.segBuffer.masks)
        self.DCplot.setOverlay(img)
        self.Segplot.setImage(img, axes={'t': 0, 'x': 2, 'y': 1, 'c': None})
        self.Segplot.setColorMap(pg.colormap.get('viridis'))

    def changeSize(self, text):
        if not self.drawButton.isChecked():
            try:
                newsize = int(text)
                self.brushsize = newsize
            except:
                pass
        else:
            try:
                newsize = int(text)
                self.brushsize = newsize
                val = self.brushcolor
                kern = np.full((self.brushsize, self.brushsize), val)

                cen = self.brushsize % 2
                self.DCplot.currentover.setDrawKernel(
                    kern, mask=None, center=(cen, cen), mode='set')
            except:
                pass

    def changeCol(self, text):
        if not self.drawButton.isChecked():
            try:
                newsize = int(text)
                self.brushcolor = newsize
                self.bColIn.setText(str(self.brushcolor))
            except:
                pass
        else:
            try:
                newcol = int(text)
                self.brushcolor = newcol
                val = self.brushcolor

                kern = np.full((self.brushsize, self.brushsize), val)

                cen = self.brushsize % 2
                self.DCplot.currentover.setDrawKernel(
                    kern, mask=None, center=(cen, cen), mode='set')
                self.bColIn.setText(str(self.brushcolor))
            except:
                pass

    def drawMode(self, checked):
        if checked:
            print('entered draw mode')
            val = self.brushcolor
            kern = np.full((self.brushsize, self.brushsize), val)

            cen = self.brushsize % 2
            self.DCplot.currentover.setDrawKernel(
                kern, mask=None, center=(cen, cen), mode='set')

        else:
            self.DCplot.currentover.drawKernel = None
            self.Segplot.imageItem.drawKernel = None
            self.DCplot.updateImage()
            print('exited draw mode')

    def opacityChanged(self, value):
        self.DCplot.currentover.setOpacity(value/1000)

    def setToolbar(self):
        self.toolbar = QtWidgets.QToolBar()
        self.drawButton = QtWidgets.QAction("Draw on images", self)
        self.drawButton.setCheckable(True)
        self.drawButton.triggered.connect(self.drawMode)
        self.toolbar.addAction(self.drawButton)

        self.bSize = QtWidgets.QLabel()
        self.bSize.setText('Brush size')
        self.toolbar.addWidget(self.bSize)
        self.bSizeIn = QtWidgets.QLineEdit(str(self.brushsize))
        self.bSizeIn.setMaxLength(4)
        self.bSizeIn.setMaximumWidth(50)
        self.bSizeIn.textChanged.connect(self.changeSize)
        self.toolbar.addWidget(self.bSizeIn)

        self.bCol = QtWidgets.QLabel()
        self.bCol.setText('Brush color')
        self.toolbar.addWidget(self.bCol)
        self.bColIn = QtWidgets.QLineEdit(str(self.brushcolor))
        self.bColIn.setMaxLength(4)
        self.bColIn.setMaximumWidth(50)
        self.bColIn.textChanged.connect(self.changeCol)
        self.toolbar.addWidget(self.bColIn)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.opacityChanged)
        self.toolbar.addWidget(self.slider)

        self.resetImgs = QtWidgets.QAction("Reset Images", self)
        self.resetImgs.triggered.connect(self.resetImages)
        self.toolbar.addAction(self.resetImgs)

        self.checkImgs = QtWidgets.QAction("Checkpoint Images", self)
        self.checkImgs.triggered.connect(self.checkpoint)
        self.toolbar.addAction(self.checkImgs)
        
        self.deleteImg = QtWidgets.QAction("Delete current image", self)
        self.deleteImg.triggered.connect(self.deleteImage)
        self.toolbar.addAction(self.deleteImg)

class OverlayImage(pg.ImageView):
    sigChangedDefaultCol = QtCore.Signal(object)
    sigLabelDeleted = QtCore.Signal(object, object)
    sigClickedImage = QtCore.Signal(object, object)
    sigRemapLabels = QtCore.Signal(object,object,object)
    def __init__(self, parent):

        super().__init__()
        self.parent = parent
        self.has_overlay = False
        # store cursor positions in scene and global
        self.cursor_position = QtCore.QPoint(0, 0)
        self.global_pos = QtCore.QPoint(0, 0)

        # connect mouseMove and time change signals to display tooltip with px info
        self.imageItem.scene().sigMouseMoved.connect(self.mouseMove)
        self.sigTimeChanged.connect(self.tChange)
        self.view.setMenuEnabled(False)
        # self.view.mouseDragEvent = types.MethodType(mouseDragEvent,self.view)
        self.menu = QtWidgets.QMenu()
        self.contextChange = QtWidgets.QAction('Change label')
        self.contextChange.triggered.connect(self.changeLabel)

        self.contextDelete = QtWidgets.QAction('Delete label')
        self.contextDelete.triggered.connect(self.deleteLabel)
        self.donothing = QtWidgets.QAction('Do nothing')
        self.donothing.triggered.connect(self.donoth)
        self.menu.addAction(self.contextDelete)
        self.menu.addAction(self.contextChange)
        self.menu.addAction(self.donothing)
        self.view.mouseDragEvent = self.customMouseDragEvent

    def customMouseDragEvent(self, ev, axis=None):
        # Get the original method
        original_method = pg.ViewBox.mouseDragEvent.__get__(
            self.view, pg.ViewBox)

        # Call the original method only for non-right button events
        if ev.button() != QtCore.Qt.RightButton:
            original_method(ev, axis)
        else:
            ev.accept()  # Accept right button events but don't process them

    def rescaleLabels(self):
        # make range of unique data same size as uniques in this image
        newlabels = np.arange(len(np.unique(self.currentover.image)))
        oldlabels = np.unique(self.currentover.image)
        for i in range(newlabels.shape[0]):
            self.currentover.image[self.currentover.image == oldlabels[i]] = newlabels[i]
            
        self.sigRemapLabels.emit(oldlabels,newlabels,self.currentIndex)
        self.updateImage()

    def tChange(self, ind, times):
        # get array indexes for mouse
        row, col = int(self.cursor_position.x()), int(self.cursor_position.y())

        # if not in array do nothing
        if not ((0 <= row < self.currentover.image.shape[0]) and (0 <= col < self.currentover.image.shape[1])):
            return
        # else display tooltip; can change tobe only on arrows to reduce bugs, but works for now
        message = f'X = {self.cursor_position.x():.1f};\nY = {self.cursor_position.y():.1f};\n' + \
            f'Value = {self.currentover.image[row,col]};'
        QtWidgets.QToolTip.showText(self.global_pos, message)

    def mouseMove(self, abs_pos):
        # same as above, but also store mouse pos in local variable
        pos = self.imageItem.mapFromScene(abs_pos)
        row, col = int(pos.y()), int(pos.x())

        if ((0 <= row < self.currentover.image.shape[0]) and (0 <= col < self.currentover.image.shape[1])):
            message = f'X = {row};\nY = {col};\n' + \
                f'Value = {self.currentover.image[row,col]};'
            self.setToolTip(message)

        else:
            self.setToolTip('')
        global_pos = QtCore.QPoint(int(abs_pos.x()), int(abs_pos.y()))
        self.global_pos = self.mapToGlobal(global_pos)
        self.cursor_position = pos

    def setOverlay(self, overlay):
        if self.has_overlay:
            self.overlay = overlay.copy()
            self.currentover.updateImage(self.overlay[self.currentIndex, :, :])
            self.updateImage()
        else:
            self.overlay = overlay.copy()
            self.currentover = pg.ImageItem(
                self.overlay[self.currentIndex, :, :])
            cmap = pg.colormap.get('viridis')
            self.currentover.setColorMap(cmap)
            self.addItem(self.currentover)
            self.currentover.setOpacity(0.5)
            self.currentover.setZValue(10)
            self.has_overlay = True
            self.updateImage()

    def updateImage(self, autoHistogramRange=True):
        super().updateImage()
        if self.has_overlay:
            self.currentover.updateImage(
                self.overlay[self.currentIndex, :, :], autoHistogramRange=True)
            cmap = pg.colormap.get('viridis')
            self.currentover.setColorMap(cmap)
            self.currentover.setLevels((0, np.max(self.currentover.image)))
            newcol = str(len(np.unique(self.currentover.image.flatten())))
            self.sigChangedDefaultCol.emit(newcol)

    def contextMenuEvent(self, event):

        point = self.imageItem.mapFromScene(event.pos())
        self.global_position = QtCore.QPoint(int(point.x()), int(point.y()))

        context_position = event.globalPos()
        self.menu.popup(context_position)
        self.resetMouseState()
        print('Ended execution of menu')

    def resetMouseState(self):

        self.view.mouseIsPressed = False

        # Reset the state of other related objects
        if hasattr(self.view, 'scene'):
            scene = self.view.scene()
            if hasattr(scene, 'clickEvents'):
                print('resetclicks')
                scene.clickEvents = []

        # Force an update
        self.view.update()

    def donoth(self):
        print('do nothing')

    def deleteLabel(self):

        pos = self.global_position

        row, col = int(pos.y()), int(pos.x())

        if ((0 <= row < self.currentover.image.shape[0]) and (0 <= col < self.currentover.image.shape[1])):
            val = self.currentover.image[row, col]
            self.currentover.image[self.currentover.image == val] = 0
            self.rescaleLabels()
            self.currentover.updateImage(self.currentover.image)
            self.sigLabelDeleted.emit(self.currentIndex, val)
            print(f'Deleting label {val}')
        else:
            print("No label to delete")
            return
        self.updateImage()
        print('delete label')

    def changeLabel(self):

        pos = self.global_position

        row, col = int(pos.y()), int(pos.x())

        if ((0 <= row < self.currentover.image.shape[0]) and (0 <= col < self.currentover.image.shape[1])):
            val = self.currentover.image[row, col]
            print(f'Changing label {val}')
            self.currentover.image[self.currentover.image ==
                                   val] = self.parent.brushcolor
            self.rescaleLabels()
            self.currentover.updateImage(self.currentover.image)
        else:
            print("No label to change")
            return
        self.updateImage()
        print('Change label')

    def mouseDoubleClickEvent(self, event):
        pos = self.cursor_position

        row, col = int(pos.y()), int(pos.x())

        if ((0 <= row < self.currentover.image.shape[0]) and (0 <= col < self.currentover.image.shape[1])):
            if event.button() == QtCore.Qt.LeftButton:
                print("Clicked on smth")
                self.sigClickedImage.emit(
                    self.currentIndex, self.currentover.image[row, col])


class TrackWindow(QtWidgets.QMainWindow):
    sigKeyPressed = QtCore.Signal(object)

    def __init__(self, segBuffer, analysisBuffer, stabilize=True):
        super().__init__()
        self.segBuffer = segBuffer
        self.analysisBuffer = analysisBuffer
        # stabilize images in buffers
        if stabilize:
            if len(self.segBuffer.images) == len(self.analysisBuffer.ACs):
                print(
                    f"Analysis buffer is in sync with segBuffer: {len(self.segBuffer.images)}")
                newDC, segs, newAC = stabilize_images(
                    self.segBuffer.images, self.segBuffer.masks, self.analysisBuffer.ACs)
                segs = [s.astype(int) for s in segs]
    
                self.segBuffer.masks = segs
                for DC in newDC:
                    minval = np.mean(DC[DC.astype(float) != float(0)])
                    DC[DC.astype(float) == float(0)] = minval
                self.segBuffer.images = newDC
                self.analysisBuffer.DCs = newDC
                self.analysisBuffer.ACs = newAC
            else:  # segBuffer and analysisBuffer are not synced
                print(
                    f"Analysis buffer is NOT in sync with segBuffer: {len(self.analysisBuffer.ACs),len(self.segBuffer.images)}")
                newDC, segs = stabilize_images(
                    self.segBuffer.images, self.segBuffer.masks)
    
                segs = [s.astype(int) for s in segs]
                self.segBuffer.masks = segs
                for DC in newDC:
                    minval = np.mean(DC[DC.astype(float) != float(0)])
                    DC[DC.astype(float) == float(0)] = minval
                self.segBuffer.images = newDC

        self.wid = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self.wid)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(splitter)
        self.brushsize = 3
        self.brushcolor = 0
        self.setToolbar()

        self.DCplot = OverlayImage(self)
        self.DCplot.sigChangedDefaultCol.connect(self.changeCol)
        self.GraphPlot = pg.PlotWidget()
        self.Graph = Graph()
        self.Graph.sigPointSelected.connect(self.highlightCell)
        self.Graph.sigPanView.connect(self.panView)
        self.GraphPlot.addItem(self.Graph)
        splitter.addWidget(self.DCplot)
        splitter.addWidget(self.GraphPlot)
        self.imagenet = obtain_network(self.segBuffer.masks.copy())
        if isinstance(self.imagenet, int):
            idx = self.imagenet
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(
                "Empty segmentation image contained at index {idx} in file {self.analysisBuffer.names[idx}")
            msg.setInformativeText(
                'Retry tracking with valid segmentations!')
            msg.setWindowTitle("TypeError")
            msg.exec_()
            self.close()
        self.segBuffer.imagenet = deepcopy(self.imagenet)
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                           text=text, size=15, meta=meta)
        self.DCplot.sigClickedImage.connect(self.Graph.cellClicked)
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        splitter.addWidget(self.DCplot)
        splitter.addWidget(self.GraphPlot)
        img = np.asarray(self.segBuffer.masks)

        self.DCplot.setOverlay(img)

        # set up links, clear unneeded UI

        self.DCplot.ui.roiBtn.hide()
        self.DCplot.ui.menuBtn.hide()
        self.DCplot.view.setMenuEnabled(False)

        self.layout.setMenuBar(self.toolbar)
        self.wid.setLayout(self.layout)
        self.setCentralWidget(self.wid)
        self.showMaximized()

        print(f'minvalue in DC: {np.min(self.DCplot.imageItem.image)}')
        self.DCplot.autoRange()
        self.highlight = None
        self.DCplot.sigTimeChanged.connect(self.switchoffHighlight)
        self.Graph.sigPointDeselected.connect(self.switchoffHighlight)
        self.GraphPlot.keyPressEvent = self.customKeyPress
        self.sigKeyPressed.connect(self.Graph.keyPressEvent)
        self.Graph.sigNodesLinked.connect(self.linkNodes)
        self.Graph.sigNodeDisconnected.connect(self.unlinkNode)

    def rescaledLabels(self,oldlabels,newlabels,currIdx):
        self.imagenet.reassignLabels(oldlabels,newlabels,currIdx)
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                           text=text, size=15, meta=meta)
        self.switchoffHighlight(0)
    def changeText(self, idx):
        match idx:
            case 0:
                text = str(0.0)
            case 1:
                text = str(100)
            case 2:
                text = str(1.1)
        self.convIn.setText(text)

    def filterNodes(self):
        overlays = self.DCplot.overlay
        choice = self.combobox.currentIndex()
        
        match choice:
            case 0:  # Convexity

                remove_list = self.imagenet.filterByConvexity(
                    float(self.convIn.text()))
            case 1:  # Size
                remove_list = self.imagenet.filterByArea(
                    float(self.convIn.text()))
            case 2:  # AR
                remove_list = self.imagenet.filterByAspectRatio(
                    float(self.convIn.text()))
        
        tic = tm.time()
        for node in remove_list:
            k = overlays[int(node[0]), :, :]
            k[k == float(node[1])] = 0
            overlays[int(node[0]), :, :] = k
        print(f'Removing nodes from array took {round(tm.time()-tic,3)}s')
        for i in range(overlays.shape[0]):
            self.DCplot.setCurrentIndex(i)
            self.DCplot.rescaleLabels()
        
        self.DCplot.setCurrentIndex(0)
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        self.DCplot.setOverlay(overlays)
        self.DCplot.updateImage()
        self.imagenet = obtain_network(
            [overlays[i, :, :] for i in range(overlays.shape[0])])
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                           text=text, size=15, meta=meta)
        self.switchoffHighlight(0)

    def openVisual(self):
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            lineage = self.imagenet.getLineageDict()
            ACs = self.analysisBuffer.ACs[:]
            DCs = self.analysisBuffer.DCs[:]
            labels = self.segBuffer.masks[:]
            times = self.analysisBuffer.abstimes[:]
            print(times)
            times = list(sorted(times))
            times = np.asarray([t-times[0] for t in times]).astype(int)
            tic = tm.time()
            cell_data = process_cells(lineage, ACs, DCs, labels, times)
            print(f'Calculating per cell took {round(tm.time()-tic,3)} for {len(cell_data.keys())} cells.')
            print(cell_data[list(cell_data.keys())[0]])
            print(times)
            self.vis = CellViewer(cell_data,ACs,DCs,times)
            self.vis.show()
            
    def openTimeSeries(self):
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            lineage = self.imagenet.getLineageDict()
            ACs = self.analysisBuffer.ACs[:]
            DCs = self.analysisBuffer.DCs[:]
            labels = self.segBuffer.masks[:]
            times = self.analysisBuffer.abstimes[:]
            print(times)
            times = list(sorted(times))
            times = np.asarray([t-times[0] for t in times]).astype(int)
            tic = tm.time()
            cell_data = process_cells(lineage, ACs, DCs, labels, times)
            print(f'Calculating per cell took {round(tm.time()-tic,3)} for {len(cell_data.keys())} cells.')
            print(cell_data[list(cell_data.keys())[0]])
            print(times)
            self.vis = DataVisualizerApp(cell_data)
            self.vis.show()
            
    def panView(self, x, y):
        self.GraphPlot.getViewBox().translateBy(x=x, y=y)

    def linkNodes(self, current, target):
        if current[0] == target[0]:
            return
        if current[0] < target[0]:
            self.imagenet.linknodes(current, target)
        else:
            self.imagenet.linknodes(target, current)
        self.imagenet.makeTree()
        self.imagenet.virtualizeCoords()
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                           text=text, size=15, meta=meta)
        self.Graph.updateGraph()

    def unlinkNode(self, nodepos,direction):
        self.imagenet.unlinknode(nodepos,direction)
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                           text=text, size=15, meta=meta)
        self.Graph.updateGraph()

    def customKeyPress(self, ev):
        # Get the original method
        original_method = pg.PlotWidget.keyPressEvent.__get__(
            self.GraphPlot, pg.PlotWidget)

        # Call the original method only for non-right button events
        self.sigKeyPressed.emit(ev)
        original_method(ev)

    def switchoffHighlight(self, t):
        if self.highlight is not None:
            print("Clear highlight")
            self.highlight.clear()
            self.highlight.update()
            self.DCplot.update()

    def highlightCell(self, t, label):

        # no guards! trust
        self.DCplot.setCurrentIndex(int(t))
        outlineimg = self.DCplot.currentover.image.copy()
        outlineimg[outlineimg != label] = 0
        outlineimg = outlineimg/label
        contour, _ = cv2.findContours(outlineimg.astype(
            'uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contourx = contour[0][:, 0, 0]
        contoury = contour[0][:, 0, 1]
        contourx = np.append(contourx, contourx[0])
        contoury = np.append(contoury, contoury[0])
        pen = pg.mkPen('r', width=5)
        paint = pg.PlotCurveItem(x=contourx, y=contoury, pen=pen)
        if self.highlight is None:
            self.highlight = paint
            self.DCplot.addItem(self.highlight)
            self.highlight.setZValue(20)

        else:
            self.highlight.updateData(x=contourx, y=contoury, pen=pen)
            self.highlight.setZValue(20)

    def resetImages(self):
        currentIdx = self.DCplot.currentIndex
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        img = np.asarray(self.segBuffer.masks)
        self.DCplot.setOverlay(img)
        self.imagenet = deepcopy(self.segBuffer.imagenet)
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                           text=text, size=15, meta=meta)
        self.DCplot.setCurrentIndex(min(currentIdx,img.shape[0]-1))
        
    def recomputeNet(self):
        imgs = self.DCplot.overlay
        imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        self.imagenet = obtain_network(imgs)
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                           text=text, size=15, meta=meta)
        
    def checkpoint(self):
        imgs = self.DCplot.overlay
        imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        self.segBuffer.masks = imgs
        self.segBuffer.imagenet = deepcopy(self.imagenet)
        
    def deleteImage(self):
        currentIdx = self.DCplot.currentIndex
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            self.segBuffer.images.pop(currentIdx)
            self.segBuffer.masks.pop(currentIdx)
            self.analysisBuffer.ACs.pop(currentIdx)
            self.analysisBuffer.DCs.pop(currentIdx)
            self.analysisBuffer.names.pop(currentIdx)
            self.analysisBuffer.times.pop(currentIdx)
            self.analysisBuffer.abstimes.pop(currentIdx)
        self.resetImages()
    def changeSize(self, text):
        if not self.drawButton.isChecked():
            try:
                newsize = int(text)
                self.brushsize = newsize
            except:
                pass
        else:
            try:
                newsize = int(text)
                self.brushsize = newsize
                val = self.brushcolor
                kern = np.full((self.brushsize, self.brushsize), val)

                cen = self.brushsize % 2
                self.DCplot.currentover.setDrawKernel(
                    kern, mask=None, center=(cen, cen), mode='set')
            except:
                pass

    def changeCol(self, text):
        if not self.drawButton.isChecked():
            try:
                newsize = int(text)
                self.brushcolor = newsize
                self.bColIn.setText(str(self.brushcolor))
            except:
                pass
        else:
            try:
                newcol = int(text)
                self.brushcolor = newcol
                val = self.brushcolor

                kern = np.full((self.brushsize, self.brushsize), val)

                cen = self.brushsize % 2
                self.DCplot.currentover.setDrawKernel(
                    kern, mask=None, center=(cen, cen), mode='set')
                self.bColIn.setText(str(self.brushcolor))
            except:
                pass

    def drawMode(self, checked):
        if checked:
            print('entered draw mode')
            val = self.brushcolor
            kern = np.full((self.brushsize, self.brushsize), val)

            cen = self.brushsize % 2
            self.DCplot.currentover.setDrawKernel(
                kern, mask=None, center=(cen, cen), mode='set')

        else:
            self.DCplot.currentover.drawKernel = None
            self.DCplot.updateImage()
            print('exited draw mode')

    def opacityChanged(self, value):
        self.DCplot.currentover.setOpacity(value/1000)

    def setToolbar(self):
        self.toolbar = QtWidgets.QToolBar()
        self.drawButton = QtWidgets.QAction("Draw on images", self)
        self.drawButton.setCheckable(True)
        self.drawButton.triggered.connect(self.drawMode)
        self.toolbar.addAction(self.drawButton)

        self.bSize = QtWidgets.QLabel()
        self.bSize.setText('Brush size')
        self.toolbar.addWidget(self.bSize)
        self.bSizeIn = QtWidgets.QLineEdit(str(self.brushsize))
        self.bSizeIn.setMaxLength(4)
        self.bSizeIn.setMaximumWidth(50)
        self.bSizeIn.textChanged.connect(self.changeSize)
        self.toolbar.addWidget(self.bSizeIn)

        self.bCol = QtWidgets.QLabel()
        self.bCol.setText('Brush color')
        self.toolbar.addWidget(self.bCol)
        self.bColIn = QtWidgets.QLineEdit(str(self.brushcolor))
        self.bColIn.setMaxLength(4)
        self.bColIn.setMaximumWidth(50)
        self.bColIn.textChanged.connect(self.changeCol)
        self.toolbar.addWidget(self.bColIn)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.opacityChanged)
        self.toolbar.addWidget(self.slider)

        self.resetImgs = QtWidgets.QAction("Reset Images", self)
        self.resetImgs.triggered.connect(self.resetImages)
        self.toolbar.addAction(self.resetImgs)

        self.checkImgs = QtWidgets.QAction("Checkpoint Images", self)
        self.checkImgs.triggered.connect(self.checkpoint)
        self.toolbar.addAction(self.checkImgs)
        
        self.remakeNet = QtWidgets.QAction("Recompute network", self)
        self.remakeNet.triggered.connect(self.recomputeNet)
        self.toolbar.addAction(self.remakeNet)
        
        self.deleteImg = QtWidgets.QAction("Delete current image", self)
        self.deleteImg.triggered.connect(self.deleteImage)
        self.toolbar.addAction(self.deleteImg)
        
        self.openTS = QtWidgets.QAction("Open Cell Time Series", self)
        self.openTS.triggered.connect(self.openTimeSeries)
        self.toolbar.addAction(self.openTS)
        
        self.openVis = QtWidgets.QAction("Open Cell Visualiser", self)
        self.openVis.triggered.connect(self.openVisual)
        self.toolbar.addAction(self.openVis)
        

        self.filter = QtWidgets.QAction("Filter Cells", self)
        self.filter.setCheckable(False)
        self.filter.triggered.connect(self.filterNodes)
        self.toolbar.addAction(self.filter)

        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(["Convexity", "Size", "Aspect Ratio"])
        self.combobox.currentIndexChanged.connect(self.changeText)
        self.toolbar.addWidget(self.combobox)

        self.convIn = QtWidgets.QLineEdit(str(0.5))
        self.convIn.setMaxLength(4)
        self.convIn.setMaximumWidth(50)
        self.toolbar.addWidget(self.convIn)


class Graph(pg.GraphItem):
    sigPointSelected = QtCore.Signal(object, object)
    sigPointDeselected = QtCore.Signal(object)
    sigPanView = QtCore.Signal(object, object)
    sigNodesLinked = QtCore.Signal(object, object)
    sigNodeDisconnected = QtCore.Signal(object,object)
    sigContextMenuOpened = QtCore.Signal()

    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        self.selectedIndices = None
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)
        self.scatter.sigHovered.connect(self.on_scatter_hover)
        self.linking = False
        self.menu = None

    def getMenu(self):
        menu = QtWidgets.QMenu()
        self.contextLink = QtWidgets.QAction('Link with next selected node')
        self.contextLink.triggered.connect(self.enableLinks)
        self.contextUnlinkLeft = QtWidgets.QAction(
            'Break left-side links')
        self.contextUnlinkLeft.triggered.connect(self.disconnectNodeLeft)
        self.contextUnlinkRight = QtWidgets.QAction(
            'Break right-side links')
        self.contextUnlinkRight.triggered.connect(self.disconnectNodeRight)
        self.getInfo()
        menu.addAction(self.contextLink)
        menu.addAction(self.contextUnlinkLeft)
        menu.addAction(self.contextUnlinkRight)
        menu.addSeparator()
        menu.addAction('Label: '+self.info[0])
        menu.addAction('Convexity: '+self.info[2])
        menu.addAction('Size: '+self.info[1])
        menu.addAction('Aspect Ratio: '+self.info[3])
        return menu

    def getInfo(self):
        self.info = [str(self.metadata[self.selectedIndices][i])
                     if self.selectedIndices is not None else '---' for i in range(1, 5)]

    def cellClicked(self, t, l):
        for i, el in enumerate(self.metadata):
            if el[0] == int(t) and el[1] == int(l):
                index = i
        self.selectedIndices = index
        self.updateGraph()
        self.sigPointSelected.emit(
            self.metadata[index][0], self.metadata[index][1])
        print("Updated graph")

    def enableLinks(self):
        self.linking = True

    def disconnectNodeLeft(self):
        nodepos = self.metadata[self.selectedIndices]
        self.sigNodeDisconnected.emit(nodepos,'l')
    def disconnectNodeRight(self):
        nodepos = self.metadata[self.selectedIndices]
        self.sigNodeDisconnected.emit(nodepos,'r')

    def keyPressEvent(self, ev):
        if ev.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_Left, QtCore.Qt.Key_Right] and self.selectedIndices is not None:
            currentpos = self.data['pos'][self.selectedIndices]
            oldres = self.metadata[self.selectedIndices]
            match ev.key():
                case QtCore.Qt.Key_Up:
                    newfocus = self.scatter.pointsAt(
                        QtCore.QPointF(currentpos[0], currentpos[1]+1))
                    if len(newfocus) != 0:
                        newindex = newfocus[0].data()[0]
                        self.selectedIndices = newindex
                        res = self.metadata[newindex]
                        self.updateGraph()
                        self.sigPointSelected.emit(res[0], res[1])
                        newpos = self.data['pos'][self.selectedIndices]
                        self.sigPanView.emit(
                            newpos[0]-currentpos[0], newpos[1]-currentpos[1])
                case QtCore.Qt.Key_Down:
                    newfocus = self.scatter.pointsAt(
                        QtCore.QPointF(currentpos[0], currentpos[1]-1))
                    if len(newfocus) != 0:

                        newindex = newfocus[0].data()[0]
                        self.selectedIndices = newindex
                        res = self.metadata[newindex]
                        self.updateGraph()
                        self.sigPointSelected.emit(res[0], res[1])
                        newpos = self.data['pos'][self.selectedIndices]
                        self.sigPanView.emit(
                            newpos[0]-currentpos[0], newpos[1]-currentpos[1])
                case QtCore.Qt.Key_Right:
                    newindex = np.argwhere(
                        self.data['adj'][:, 0] == self.selectedIndices)[0]
                    if len(newindex) != 0:
                        newindex = newindex[0]
                        newindex = self.data['adj'][newindex, 1]
                        self.selectedIndices = newindex
                        res = self.metadata[newindex]
                        self.updateGraph()
                        self.sigPointSelected.emit(res[0], res[1])
                        newpos = self.data['pos'][self.selectedIndices]
                        self.sigPanView.emit(
                            newpos[0]-currentpos[0], newpos[1]-currentpos[1])
                case QtCore.Qt.Key_Left:
                    newindex = np.argwhere(
                        self.data['adj'][:, 1] == self.selectedIndices)[0]
                    if len(newindex) != 0:
                        newindex = newindex[0]
                        newindex = self.data['adj'][newindex, 0]
                        self.selectedIndices = newindex
                        res = self.metadata[newindex]
                        self.updateGraph()
                        self.sigPointSelected.emit(res[0], res[1])
                        newpos = self.data['pos'][self.selectedIndices]
                        self.sigPanView.emit(
                            newpos[0]-currentpos[0], newpos[1]-currentpos[1])

    def contextMenuEvent(self, event):
        self.menu = self.getMenu()
        self.menu.popup(event.screenPos())
        self.sigContextMenuOpened.emit()
        print('Ended execution of menu')

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.metadata = kwds.pop('meta', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def updateGraph(self, *args):

        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        if self.selectedIndices is not None:
            self.scatter.points()[self.selectedIndices].setPen(pg.mkPen('r'))
            self.scatter.points()[self.selectedIndices].setBrush(
                pg.mkBrush('r'))

    def on_scatter_hover(self, item, points, _evt):
        if not len(points):
            item.setToolTip(None)
            return
        data = points[0].data()
        item.setToolTip(data)

    def clicked(self, obj, pts):
        if self.linking and self.selectedIndices is not None:  # if linking is set and a point is selected
            ind = pts[0].data()[0]
            res_destination = self.metadata[ind]
            res_current = self.metadata[self.selectedIndices]
            self.linking = False
            self.sigNodesLinked.emit(res_current, res_destination)
        else:
            ind = pts[0].data()[0]
            if ind != self.selectedIndices:
                self.selectedIndices = ind

                res = self.metadata[ind]
                self.updateGraph()
                self.sigPointSelected.emit(res[0], res[1])
            else:
                self.selectedIndices = None
                self.sigPointDeselected.emit(0)
                self.updateGraph()


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
        self.ACs.append(AC)
        self.DCs.append(DC)
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
        elif ext.upper() == 'PM':
            if len(timestring) == 5:
                # highly unlikely to be 10 PM or after, fix if possible
                hours = int(timestring[0])+12
                minutes = int(timestring[1:3])
            else:
                # highly unlikely to be 10 PM or after, fix if possible
                hours = int(timestring[0:2])
                minutes = int(timestring[2:4])
        time = (hours, minutes)
        abstime = 60*hours+minutes
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

        self.images.append(img)

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


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName('AC analysis')
    imageViewer = MainWindow()
    imageViewer.showMaximized()
    sys.exit(app.exec_())
