import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets, QtCore
from copy import deepcopy
from AnalysisGUI.CellImageViewer import CellViewer
import os
import cv2
import time as tm
from AnalysisGUI.image_holder import ImageHolder
from AnalysisGUI.buffers import SegmentationBuffer, AnalysisBuffer
from AnalysisGUI.segmentation import SegmentWindow
from AnalysisGUI.tracking import TrackWindow
from AnalysisGUI.widgets import DICWidget,DCWidget,ACWidget,Signals,FileExplorer
pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('background', 'k')





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

