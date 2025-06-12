import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import os
from copy import deepcopy

pg.setConfigOption('imageAxisOrder', 'row-major')


# function to artificially add channels to images/list of images,
# as they are monochrome 16-bit

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
        self.setExpanded(self.model.index(QtCore.QDir.rootPath()), True)
        self.setExpanded(self.model.index(os.path.join(QtCore.QDir.rootPath(),"Users")), True)
        self.setExpanded(self.model.index(QtCore.QDir.homePath()), True)
        self.resizeColumnToContents(0)
        self.resizeColumnToContents(0)

# controls viewing of raw images


class DICWidget(pg.ImageView):
    def __init__(self, parent, imagesource):
        super().__init__(discreteTimeLine=True, levelMode='mono')
        self.parent = parent
        self.imageSource = imagesource  # link to ImageHolder to update analysis
        self.setImage(self.imageSource.raws,axes={'t': 0, 'x': 2, 'y': 1, 'c': None}, xvals=np.linspace(
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
            f'Value = {self.image[ind,row,col]};'
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
        # recompute t axis to hold time in seconds in absolute terms
        self.setImage(DICdata, axes={'t': 0, 'x': 2, 'y': 1, 'c': None}, xvals=np.linspace(
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
        self.updateButton.triggered.connect(self.updateAnalysis)
        self.toolbar.addAction(self.updateButton)
        self.resetButton = QtWidgets.QAction("Reset Analysis", self)
        self.resetButton.setStatusTip("Reset limits to 0,end")
        self.resetButton.triggered.connect(self.resetAnalysis)
        self.toolbar.addAction(self.resetButton)

        self.freqLabel = QtWidgets.QAction("Frequency [Hz]:")
        self.freqLabel.setStatusTip("Set frequency in Hz")
        self.toolbar.addAction(self.freqLabel)
        self.freqIn = QtWidgets.QLineEdit(str(1))
        self.freqIn.setMaxLength(4)
        self.freqIn.setMaximumWidth(50)
        self.toolbar.addWidget(self.freqIn)
        
        self.interpButton = QtWidgets.QCheckBox("Interpolate", self)
        self.interpButton.setStatusTip("Interpolate signal")
        self.interpButton.setChecked(False)
        self.toolbar.addWidget(self.interpButton)

        self.filtButton = QtWidgets.QCheckBox("Filter", self)
        self.filtButton.setStatusTip("Filter signal")
        self.filtButton.setChecked(False)
        self.toolbar.addWidget(self.filtButton)

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
        self.limits = (self.imageSource.limits[0], self.imageSource.limits[1])
        self.region.setRegion((0, self.time[-1]))
        self.region.setBounds((0, self.time[-1]))

        # make a layout
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setMenuBar(self.toolbar)
        self.setLayout(self.layout)
        self.layout.addWidget(self.signalPlot)
        self.layout.addWidget(self.FFTPlot)
        self.show()

    def resetAnalysis(self):
        # reset limits from imagesource raws: those never change, and we can get back to start
        limits = (0, int(len(self.imageSource.raws)))
        print(f'set new limits {limits}')
        frequency = 1.0
        filt = False
        interp = False
        self.freqIn.setText(str(frequency))
        self.interpButton.setChecked(interp)
        self.filtButton.setChecked(filt)

        self.imageSource.reanalyze(limits=limits,filt = filt,interp = interp, frequency=frequency,hardlimits = False)
    
    def updateAnalysis(self):
        # set limits to those enclosed by the region; minX/maxX are updated on region move
        self.updateRegions()
        limits = (int(round(self.minX*self.imageSource.framerate))+self.imageSource.limits[0], int(round(self.maxX*self.imageSource.framerate))+self.imageSource.limits[0])
        print(f'set new limits {self.limits}')
        self.imageSource.reanalyze(limits=self.limits,
                                  frequency=float(self.freqIn.text()),
                                  interp=self.interpButton.isChecked(),
                                  filt=self.filtButton.isChecked())

    def updateRegions(self):
        # here we update region limits in case we want to redo analysis
        self.minX, self.maxX = self.region.getRegion()
        minindex = np.abs( self.time- self.minX).argmin()
        maxindex = np.abs( self.time- self.maxX).argmin()
        self.limits = (minindex+self.imageSource.limits[0], maxindex+1+self.imageSource.limits[0])
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
        self.limits = (self.imageSource.limits[0], self.imageSource.limits[1])
        self.region.setRegion([self.minX, self.maxX])
        self.region.setBounds((0, self.time[-1]))
        self.freqIn.setText(str(self.imageSource.frequency))
        self.interpButton.setChecked(self.imageSource.interpolate)
        self.filtButton.setChecked(self.imageSource.filter)


# fairly standard extension of ImageView, includes tooltip px values and links to ROI-to-signal
class DCWidget(pg.ImageView):
    def __init__(self, parent, imagesource):
        super().__init__(levelMode='mono')

        self.parent = parent
        self.imageSource = imagesource
        chans = self.imageSource.DC
        self.setImage(chans,axes={'t': None, 'x': 1, 'y': 0, 'c': None})
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
        self.setImage(chans,axes={'t': None, 'x': 1, 'y': 0, 'c': None})
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
        
        self.setImage(chans,axes={'t': None, 'x': 1, 'y': 0, 'c': None})

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
        self.setImage(ACdata,axes={'t': None, 'x': 1, 'y': 0, 'c': None})
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
        
        self.setImage(chans,axes={'t': None, 'x': 1, 'y': 0, 'c': None})
        print('Set in AC')


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
                
class Graph(pg.GraphItem):
    sigPointSelected = QtCore.Signal(object, object)
    sigPointDeselected = QtCore.Signal(object)
    sigPanView = QtCore.Signal(object, object)
    sigNodesLinked = QtCore.Signal(object, object)
    sigNodeDisconnected = QtCore.Signal(object,object)
    sigDeletedLineage = QtCore.Signal(object)
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
        self.contextDeleteLineage = QtWidgets.QAction('Delete lineage downstream')
        self.contextDeleteLineage.triggered.connect(self.deleteLineageDownstream)

        self.getInfo()
        menu.addAction(self.contextLink)
        menu.addAction(self.contextUnlinkLeft)
        menu.addAction(self.contextUnlinkRight)
        menu.addAction(self.contextDeleteLineage)
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
    
    def deleteLineageDownstream(self):
        nodename = self.text[self.selectedIndices]
        self.sigDeletedLineage.emit(nodename)
        self.selectedIndices = None
        self.updateGraph()
        self.sigPointDeselected.emit(0)
        print(f'Deleted lineage {nodename}')

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
                    try:
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
                    except IndexError:
                        pass
                case QtCore.Qt.Key_Left:
                    try:
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
                    except IndexError:
                        pass

    def contextMenuEvent(self, event):
        event.accept()
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