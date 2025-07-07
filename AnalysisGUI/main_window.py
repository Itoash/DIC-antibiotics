import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets, QtCore
from copy import deepcopy
from AnalysisGUI.cell_viewer import CellViewer
import os
import cv2
import time as tm
from AnalysisGUI.image_holder import ImageHolder
from AnalysisGUI.buffers import SegmentationBuffer, AnalysisBuffer
from AnalysisGUI.segmentation import SegmentWindow
from AnalysisGUI.tracking import TrackWindow
from AnalysisGUI.widgets import DICWidget,DCWidget,ACWidget,Signals,FileExplorer
import sif_parser as sp
pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('background', 'k')

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the AC analysis GUI.

    This class manages the main interface, including loading, processing,
    and displaying image data, as well as segmentation and tracking subwindows.
    """

    def __init__(self):
        """
        Initialize the main window and its components.
        """
        super().__init__()
        # Initialize image holder with random data
        self.imageData = ImageHolder(self,
                                     np.asarray([np.random.uniform(0, 1, (300, 300))
                                                 for i in range(400)]))
        # Initialize buffers for segmentation and analysis
        self.segBuffer = SegmentationBuffer()
        self.analysisImages = AnalysisBuffer(self.segBuffer)
        # Create dock area for plots and widgets
        self.docks = AnalysisArea(self)
        self.segmentor = None
        self.tracker = None

        self.setCentralWidget(self.docks)

        # Update plots, create menus, and set window title
        self.updateAnalysis()
        self.createActions()
        self.createMenu()
        self.setWindowTitle('Analiza AC')

    def updateAnalysis(self):
        """
        Update all plots with the current image data.
        """
        if self.imageData:
            self.docks.wDIC.update()
            self.docks.wDC.update()
            self.docks.wAC.update()
            self.docks.wSig.updateSignals()

    def resetAnalysis(self):
        """
        Reset the image data and update plots.
        """
        self.imageData.reset()
        self.updateAnalysis()

    def open(self):
        """
        Open a directory containing image data.
        """
        fileName = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                              "Open File",
                                                              QtCore.QDir.currentPath())
        if fileName:
            self.loadstack(fileName)

    def loadspool(self, filename=None):
        """
        Load spooled data from a directory.

        Parameters:
        filename : str, optional
            Path to the directory containing spooled data.
        """
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
                'Please select a directory containing spooled data')
            msg.setWindowTitle("TypeError")
            msg.exec_()
            return
        spoolfile = [f for f in os.listdir(filename) if '.sifx' in f or '.ini' in f]
        if len(spoolfile)<1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No valid files!")
            msg.setInformativeText(
                'Chosen directory contains no valid spooled data (.sifx and .ini headers, .dat data files)')
            msg.setWindowTitle("TypeError")
            msg.exec_()

            return


       

        ti_c = os.path.getctime(filename)

        c_ti = tm.ctime(ti_c)

        
        ctime = tm.strptime(c_ti)
        T_stamp = tm.strftime("Spool_%Y_%m_%d_%I%M%S %p", ctime)
        images,metadata = sp.np_spool_open(filename)
        framerate = 1/metadata["CycleTime"]
        limits = (0,images.shape[0])
        self.imageData.codename = T_stamp
        self.imageData.framerate = framerate
        self.imageData.limits = limits  # set limits
        self.imageData.setRaws(images)  # change raw image data
        self.imageData.update()  # update images internally (run AC)

        self.updateAnalysis()  # update Plots
        

    def loadstack(self, filename=None):
        """
        Load a stack of images from a directory.

        Parameters:
        filename : str, optional
            Path to the directory containing image stack.
        """
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
        tic = tm.time()
        images = [cv2.imread(os.path.join(filename, name), -1)
                  for name in validfiles]
        print(f'Loading took {tm.time()-tic:.3f}s')
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
        limits = (0, len(images)-1)
        images = np.asarray(images)
        _,codename = os.path.split(filename)
        print(codename)
        self.imageData.codename = codename
        self.imageData.framerate = framerate
        self.imageData.limits = limits  # set limits
        self.imageData.setRaws(images)  # change raw image data
        self.imageData.update()  # update images internally (run AC)
        
        self.updateAnalysis()  # update Plots

    def segment(self):
        """
        Open the segmentation window.
        """
        if len(self.segBuffer.images) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No images in buffer!")
            msg.setInformativeText('Segmentation buffer contains no images')
            msg.setWindowTitle("IndexError")
            msg.exec_()
            return
        if self.segmentor is not None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Segmentor already open!")
            msg.setInformativeText('Segmentor window is already open. Please close and retrack if modifications are in order.')
            msg.setWindowTitle("SegmentorOpen")
            msg.exec_()
            return
        self.segmentor = SegmentWindow(self.segBuffer,self.analysisImages)
        self.segmentor.window_closed.connect(self.segmentor_closed)
        self.segmentor.show()

    def track(self):
        """
        Open the tracking window.
        """
        if len(self.segBuffer.images) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No images in buffer!")
            msg.setInformativeText('Segmentation buffer contains no images')
            msg.setWindowTitle("IndexError")
            msg.exec_()
            return
        if self.tracker is not None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Tracker already open!")
            msg.setInformativeText('Tracker window is already open. Please close and resegment if modifications are in order.')
            msg.setWindowTitle("TrackerOpen")
            msg.exec_()
            return
            

        self.tracker = TrackWindow(self.segBuffer, self.analysisImages,parent=self)
        self.tracker.window_closed.connect(self.tracker_closed)
        self.tracker.show()

    def tracker_closed(self):
        """
        Handle the closure of the tracker window.
        """
        self.tracker = None

    def segmentor_closed(self):
        """
        Handle the closure of the segmentor window.
        """
        self.segmentor = None

    def save(self):
        """
        Save the current image data to a file.
        """
        # save the current image to a file
        filename = QtWidgets.QFileDialog.getExistingDirectory(self, "Save Cell data" )
        if filename and 'tracker' in self.__dict__.keys() and self.tracker is not None:
            self.tracker.saveCells(filename)

    def addImages(self):
        """
        Add images to the segmentation buffer.
        """
        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open File", QtCore.QDir.homePath(), options=options)
        if filename:
            self.clearBuffer()
            ACpath = [f for f in os.listdir(filename) if 'ac' in f.lower()][0]
            DCpath = [f for f in os.listdir(filename) if 'dc' in f.lower()][0]
            segpath = [f for f in os.listdir(filename) if 'seg' in f.lower()][0]
            datapath = [f for f in os.listdir(filename) if 'data' in f.lower() and f.endswith('.csv')][0]
            with open(os.path.join(filename,datapath),'r') as file:
                lines = file.readlines()
                times = []
                for line in lines:
                    if 'times' in line:
                        times+= line.strip().split(',')[1:]
                times = [float(t) for t in times]
                times = list(set(times))
                times = list(sorted(times))
                        
            ACs = []
            DCs = []
            segs = []
            _,ACs = cv2.imreadmulti(os.path.join(filename, ACpath), ACs, -1)
            _,DCs = cv2.imreadmulti(os.path.join(filename, DCpath),DCs, -1)
            _,segs = cv2.imreadmulti(os.path.join(filename, segpath),segs, -1)
            ACs = [i.astype(float) for i in ACs]
            DCs = [i.astype(float) for i in DCs]
            segs = [i.astype(float) for i in segs]
            print(f"Loaded {len(ACs)} images")
            print(type(DCs))
            self.segBuffer.images = list(DCs)[:]
            self.segBuffer.masks = list(segs)[:]
            self.analysisImages.ACs = list(ACs)[:]
            self.analysisImages.DCs = list(DCs)[:]
            self.analysisImages.names = [str(i) for i in times]
            self.analysisImages.times = times[:]
            self.analysisImages.abstimes = times[:]
            print(f"Loaded {len(self.analysisImages.ACs)} images")
            
        
            

    def loadSpoolFolder(self):
        """
        Load a folder containing spooled data.
        """
        # get elected directory and run through some checks:
        index = self.docks.treeview.currentIndex()
        filename = self.docks.treeview.model.filePath(index)
        
        spoolfiles = [
            f for f in os.listdir(filename) if '.DS_Store' not in f
        ]
        if len(spoolfiles)<1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No valid files!")
            msg.setInformativeText(
                'Chosen directory contains no valid spooled data (.sifx and .ini headers, .dat data files)')
            msg.setWindowTitle("TypeError")
            msg.exec_()

            return
        heights = set()
        widths = set()
        for file in spoolfiles:
            ini_file = [f for f in os.listdir(os.path.join(filename,file)) if f.endswith('.ini')]
            sifx_file = [f for f in os.listdir(os.path.join(filename,file)) if f.endswith('.sifx')]
            if len(ini_file) != 1 or len(sifx_file) != 1:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText("Invalid spool files!")
                msg.setInformativeText(
                    'Each spool folder should contain exactly one .ini and one .sifx file.')
                msg.setWindowTitle("TypeError")
                msg.exec_()
                return
            with open(os.path.join(filename,file,ini_file[0]), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'AOIHeight' in line:
                        AOIheight = int(line.split('=')[1].strip())
                    if 'AOIWidth' in line:
                        AOIwidth = int(line.split('=')[1].strip())
            heights.add(AOIheight)
            widths.add(AOIwidth)
        # Check 2: if there are multiple heights or widths, display message and abort
        if len(heights) > 1 or len(widths) > 1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Different sizes!")
            msg.setInformativeText(
                'Spool files have different sizes. Select a folder with spools of the same size.')
            msg.setWindowTitle("TypeError")
            msg.exec_()
            return
        # Create a custom dialog to get number of locations and starting index
        num_locations, start_index = self.show_parameter_dialog()
        if num_locations is None or start_index is None:
            print("Dialog cancelled or invalid input")
            return
        # Check if start_index is within bounds
        if start_index < 0 or start_index >= len(spoolfiles):
            QtWidgets.QMessageBox.warning(
                self, "Index Error", "Starting index is out of bounds.")
            return
        print("Got parameters:", num_locations, start_index)
        self.clearBuffer()
        # if 'segmentor' in self.__dict__.keys():
        #     self.segmentor.close()
        # if 'tracker' in self.__dict__.keys():
        #     self.tracker.close()
        tic = tm.time()
        for i in range(start_index, len(spoolfiles), num_locations):
            if os.path.isdir(os.path.join(filename,spoolfiles[i])):
                # load images from directory
                self.loadspool(os.path.join(filename,spoolfiles[i]))
                self.addImage(resort=False)
            else:
                # something happened to directory during segmentation
                print(f"Skipping {spoolfiles[i]} as it is not a directory")
                continue
        self.analysisImages.sortByAbsTime()



    def loadDay(self):
        """
        Load a folder containing image stacks for a day.
        """
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
        
        # Create a custom dialog to get number of locations and starting index
        num_locations, start_index = self.show_parameter_dialog()
        if num_locations is None or start_index is None:
            print("Dialog cancelled or invalid input")
            return
        # Check if start_index is within bounds
        if start_index < 0 or start_index >= len(files):
            QtWidgets.QMessageBox.warning(
                self, "Index Error", "Starting index is out of bounds.")
            return
        print("Got parameters:", num_locations, start_index)
        self.clearBuffer()
        # if 'segmentor' in self.__dict__.keys():
        #     self.segmentor.close()
        # if 'tracker' in self.__dict__.keys():
        #     self.tracker.close()
        tic = tm.time()
        for i in range(start_index, len(files), num_locations):
            if os.path.isdir(files[i]):
                # load images from directory
                self.loadstack(files[i])
                self.addImage(resort=False)
            else:
                # something happened to directory during segmentation
                print(f"Skipping {files[i]} as it is not a directory")
                continue
        toc = tm.time()-tic
        print(f"Segmentation and processing took {toc}:3fs")

    def show_parameter_dialog(self):
        """
        Show a dialog to select parameters for loading data.

        Returns:
        tuple : (int, int)
            Number of locations and starting index.
        """
        # Create a custom dialog to get number of locations and starting index
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Parameters")
        dialog.setModal(True)  # Make dialog modal
        dialog.resize(300, 150)  # Set a reasonable size
        
        layout = QtWidgets.QFormLayout(dialog)
        
        # Number of locations input
        num_locations_input = QtWidgets.QSpinBox(dialog)
        num_locations_input.setMinimum(1)
        num_locations_input.setMaximum(1000)
        num_locations_input.setValue(1)
        
        # Starting index input
        start_index_input = QtWidgets.QSpinBox(dialog)
        start_index_input.setMinimum(0)
        start_index_input.setMaximum(10000)
        start_index_input.setValue(0)
        
        layout.addRow("Number of locations:", num_locations_input)
        layout.addRow("Starting index:", start_index_input)
        
        # Button box
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, 
            dialog
        )
        layout.addWidget(button_box)
        
        # Connect signals
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Execute dialog and handle result
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            num_locations = num_locations_input.value()
            start_index = start_index_input.value()
            
            
            # Process the values as needed
            print(f"Selected: {num_locations} locations starting from index {start_index}")
            
            return num_locations, start_index
        else:
            # User cancelled
            print("Dialog cancelled")
            return None, None

    def addImage(self, img=None, resort=True):
        """
        Add a single image to the segmentation buffer.

        Parameters:
        img : np.ndarray, optional
            Image data to add.
        resort : bool, optional
            Whether to resort the buffer after adding (default is True).
        """
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
        """
        Clear the segmentation and analysis buffers.
        """
        self.segBuffer.clear()
        self.analysisImages.clear()

    def about(self):
        """
        Show an "About" dialog.
        """
        # whatever this is
        QtWidgets.QMessageBox.about(self, "About Image Viewer",
                                    "<p>The <b>DIC Analyzer</b> works fine don't worry about it...</p>")

    def createActions(self):
        """
        Create actions and shortcuts for top-level commands.
        """
        # make actions and shortcuts for important top-level commands
        self.openAct = QtWidgets.QAction(
            "&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.saveAct = QtWidgets.QAction(
            "&Save...", self, shortcut="Ctrl+S", triggered=self.save)
        self.loadAct = QtWidgets.QAction(
            "&Load Stack", self, shortcut="Ctrl+L", triggered=self.loadstack)
        self.loadSpoolAct = QtWidgets.QAction(
            "&Load Spool", self, shortcut="Ctrl+P", triggered=self.loadspool)
        self.loadDayAct = QtWidgets.QAction(
            "&Load Folder of Stacks", self, shortcut="Ctrl+A", triggered=self.loadDay)
        self.loadSpoolsAct = QtWidgets.QAction(
            "&Load Folder of Spools", self, shortcut="Ctrl+R", triggered=self.loadSpoolFolder)
        
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
        """
        Close the application and clean up resources.
        """
        if self.segmentor is not None:
            self.segmentor.close()
        if self.tracker is not None:
            self.tracker.close()
        import gc
        gc.collect()
        self.close()

    def createMenu(self):
        """
        Create the menu bar and add commands.
        """
        # display commands in menubar
        self.fileMenu = QtWidgets.QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.saveAct)
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addAction(self.loadSpoolAct)
        self.fileMenu.addAction(self.loadSpoolsAct)
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
        """
        Update signal plots with the given mask.

        Parameters:
        mask : np.ndarray
            Mask for signal processing.
        """
        self.docks.wSig.updateSignals(mask)

# holds docks; can be removed and incorporated into mainwindow, but it's fine for now
class AnalysisArea(DockArea):
    """
    Dock area for managing plots and widgets.
    """

    def __init__(self, parent):
        """
        Initialize the dock area with predefined layout.

        Parameters:
        parent : object
            Reference to the parent object.
        """
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

