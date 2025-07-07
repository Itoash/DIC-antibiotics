import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from copy import deepcopy
pg.setConfigOption('imageAxisOrder', 'row-major')

class SegmentWindow(QtWidgets.QMainWindow):
    window_closed = QtCore.pyqtSignal()
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
    def closeEvent(self, event):
        self.window_closed.emit()
        event.accept()
    def deleteImage(self):
        """Delete the current image and associated data."""
        currentIdx = self.DCplot.currentIndex
        
        # Only delete if buffers are in sync
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            if len(self.segBuffer.images) == 1:

                print('Cannot delete last image')
                return
            # Remove the current image and associated data
            print('Deleting image...')
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
        self.segBuffer.masks = deepcopy(imgs)
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
        #self.sigTimeChanged.connect(self.tChange)
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