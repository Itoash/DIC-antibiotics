"""
Segmentation module for the AnalysisGUI application.

This module provides a GUI interface for interactive segmentation of cell images,
allowing users to manually edit segmentation masks, adjust brush parameters,
and manage image sequences for analysis.

Classes:
    SegmentWindow: Main window for segmentation operations
    OverlayImage: Custom ImageView widget for overlay visualization and editing
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from copy import deepcopy
pg.setConfigOption('imageAxisOrder', 'row-major')

class SegmentWindow(QtWidgets.QMainWindow):
    """
    Main window for interactive cell segmentation.
    
    This window provides tools for viewing and editing segmentation masks overlaid
    on original images. Users can draw, erase, and modify segmentation labels
    using various brush tools.
    
    Attributes:
        segBuffer: Buffer containing segmentation masks and original images
        analysisBuffer: Buffer containing analysis data (AC/DC images)
        brushsize (int): Size of the drawing brush
        brushcolor (int): Color/label value for drawing
        DCplot (OverlayImage): Main image display widget with overlay editing
        Segplot (pg.ImageView): Secondary view showing segmentation masks only
        
    Signals:
        window_closed: Emitted when the window is closed
    """
    window_closed = QtCore.pyqtSignal()
    
    def __init__(self, segBuffer, analysisBuffer):
        """
        Initialize the segmentation window.
        
        Parameters:
        -----------
        segBuffer : SegmentationBuffer
            Buffer containing segmentation masks and original images
        analysisBuffer : AnalysisBuffer
            Buffer containing analysis data (AC/DC images and metadata)
        """
        super().__init__()
        self.segBuffer = segBuffer
        self.analysisBuffer = analysisBuffer
        self.wid = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QHBoxLayout()
        self.brushsize = 3  # Default brush size for drawing
        self.brushcolor = 0  # Default brush color (background)
        
        # Set up the user interface
        self.setToolbar()
        
        # Create main image display with overlay editing capabilities
        self.DCplot = OverlayImage(self)
        self.DCplot.sigChangedDefaultCol.connect(self.changeCol)
        
        # Create secondary segmentation-only view
        self.Segplot = pg.ImageView()
        
        # Load images and masks into the displays
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        img = np.asarray(self.segBuffer.masks)
        self.DCplot.setOverlay(img)
        self.Segplot.setImage(img, axes={'t': 0, 'x': 2, 'y': 1, 'c': None})
        self.Segplot.setColorMap(pg.colormap.get('viridis'))

        # Synchronize views and clean up UI
        self._setup_view_synchronization()
        self._setup_layout()
        
        # Display the window
        self.showMaximized()
        self.DCplot.autoRange()
        self.Segplot.autoRange()
        
    def _setup_view_synchronization(self):
        """Set up synchronization between the main view and segmentation view."""
        # Link views for synchronized panning and zooming
        self.DCplot.view.setXLink(self.Segplot.view)
        self.DCplot.view.setYLink(self.Segplot.view)
        self.Segplot.view.setXLink(self.DCplot.view)
        self.Segplot.view.setYLink(self.DCplot.view)
        
        # Synchronize time slider changes
        self.DCplot.sigTimeChanged.connect(self.Segplot.setCurrentIndex)
        self.Segplot.sigTimeChanged.connect(self.DCplot.setCurrentIndex)
        
        # Hide unnecessary UI elements
        self.DCplot.ui.roiBtn.hide()
        self.DCplot.ui.menuBtn.hide()
        self.Segplot.ui.roiBtn.hide()
        self.Segplot.ui.menuBtn.hide()
        self.DCplot.view.setMenuEnabled(False)
        self.Segplot.view.setMenuEnabled(False)
        
    def _setup_layout(self):
        """Set up the main window layout."""
        self.layout.addWidget(self.DCplot)
        self.layout.addWidget(self.Segplot)
        self.layout.setMenuBar(self.toolbar)
        self.wid.setLayout(self.layout)
        self.setCentralWidget(self.wid)
    
    def closeEvent(self, event):
        """Handle window close event by emitting signal."""
        self.window_closed.emit()
        event.accept()
        
    def deleteImage(self):
        """
        Delete the current image and associated data from both buffers.
        
        This method removes the currently displayed image from both the
        segmentation buffer and analysis buffer, maintaining synchronization.
        """
        currentIdx = self.DCplot.currentIndex
        
        # Only delete if buffers are in sync
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            if len(self.segBuffer.images) == 1:
                print('Cannot delete last image')
                return
                
            # Remove the current image and associated data
            self.segBuffer.images.pop(currentIdx)
            self.segBuffer.masks.pop(currentIdx)
            self.analysisBuffer.ACs.pop(currentIdx)
            self.analysisBuffer.DCs.pop(currentIdx)
            self.analysisBuffer.names.pop(currentIdx)
            self.analysisBuffer.times.pop(currentIdx)
            self.analysisBuffer.abstimes.pop(currentIdx)
            
        self.resetImages()
            
    def resetImages(self):
        """Reset the image displays to show current buffer contents."""
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        img = np.asarray(self.segBuffer.masks)
        self.DCplot.setOverlay(img)
        self.Segplot.setImage(img, axes={'t': 0, 'x': 2, 'y': 1, 'c': None})
        self.Segplot.setColorMap(pg.colormap.get('viridis'))

    def checkpoint(self):
        """
        Save current overlay state to segmentation buffer.
        
        This creates a checkpoint of the current segmentation state,
        allowing users to save their manual edits.
        """
        # Extract current overlay data
        imgs = self.DCplot.overlay
        imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        self.segBuffer.masks = deepcopy(imgs)
        
        # Refresh the display
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        img = np.asarray(self.segBuffer.masks)
        self.DCplot.setOverlay(img)
        self.Segplot.setImage(img, axes={'t': 0, 'x': 2, 'y': 1, 'c': None})
        self.Segplot.setColorMap(pg.colormap.get('viridis'))

    def changeSize(self, text):
        """
        Change the brush size for drawing operations.
        
        Parameters:
        -----------
        text : str
            New brush size as string
        """
        if not self.drawButton.isChecked():
            try:
                newsize = int(text)
                self.brushsize = newsize
            except:
                pass
        else:
            # Update brush size and drawing kernel if in draw mode
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
        """
        Change the brush color/label for drawing operations.
        
        Parameters:
        -----------
        text : str
            New brush color as string
        """
        if not self.drawButton.isChecked():
            try:
                newsize = int(text)
                self.brushcolor = newsize
                self.bColIn.setText(str(self.brushcolor))
            except:
                pass
        else:
            # Update brush color and drawing kernel if in draw mode
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
        """
        Toggle drawing mode on/off.
        
        Parameters:
        -----------
        checked : bool
            True to enable drawing mode, False to disable
        """
        if checked:
            # Create drawing kernel with current brush parameters
            val = self.brushcolor
            kern = np.full((self.brushsize, self.brushsize), val)
            cen = self.brushsize % 2
            self.DCplot.currentover.setDrawKernel(
                kern, mask=None, center=(cen, cen), mode='set')
        else:
            # Disable drawing kernel
            self.DCplot.currentover.drawKernel = None
            self.Segplot.imageItem.drawKernel = None
            self.DCplot.updateImage()

    def opacityChanged(self, value):
        """
        Change the opacity of the segmentation overlay.
        
        Parameters:
        -----------
        value : int
            Opacity value from 0-1000 (slider range)
        """
        self.DCplot.currentover.setOpacity(value/1000)

    def setToolbar(self):
        """Set up the toolbar with drawing tools and controls."""
        self.toolbar = QtWidgets.QToolBar()
        
        # Drawing mode toggle
        self.drawButton = QtWidgets.QAction("Draw on images", self)
        self.drawButton.setCheckable(True)
        self.drawButton.triggered.connect(self.drawMode)
        self.toolbar.addAction(self.drawButton)

        # Brush size controls
        self.bSize = QtWidgets.QLabel()
        self.bSize.setText('Brush size')
        self.toolbar.addWidget(self.bSize)
        self.bSizeIn = QtWidgets.QLineEdit(str(self.brushsize))
        self.bSizeIn.setMaxLength(4)
        self.bSizeIn.setMaximumWidth(50)
        self.bSizeIn.textChanged.connect(self.changeSize)
        self.toolbar.addWidget(self.bSizeIn)

        # Brush color controls
        self.bCol = QtWidgets.QLabel()
        self.bCol.setText('Brush color')
        self.toolbar.addWidget(self.bCol)
        self.bColIn = QtWidgets.QLineEdit(str(self.brushcolor))
        self.bColIn.setMaxLength(4)
        self.bColIn.setMaximumWidth(50)
        self.bColIn.textChanged.connect(self.changeCol)
        self.toolbar.addWidget(self.bColIn)

        # Opacity slider for overlay transparency
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.opacityChanged)
        self.toolbar.addWidget(self.slider)

        # Image management buttons
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
    """
    Custom ImageView widget for displaying images with editable overlay masks.
    
    This widget extends PyQtGraph's ImageView to provide interactive editing
    capabilities for segmentation masks overlaid on original images.
    
    Signals:
        sigChangedDefaultCol: Emitted when the default color changes
        sigLabelDeleted: Emitted when a label is deleted
        sigClickedImage: Emitted when an image region is clicked
        sigRemapLabels: Emitted when labels are remapped
    """
    sigChangedDefaultCol = QtCore.Signal(object)
    sigLabelDeleted = QtCore.Signal(object, object)
    sigClickedImage = QtCore.Signal(object, object)
    sigRemapLabels = QtCore.Signal(object, object, object)
    
    def __init__(self, parent):
        """
        Initialize the OverlayImage widget.
        
        Parameters:
        -----------
        parent : QtWidgets.QWidget
            Parent widget
        """
        super().__init__()
        self.parent = parent
        self.has_overlay = False
        
        # Store cursor positions for tooltip display
        self.cursor_position = QtCore.QPoint(0, 0)
        self.global_pos = QtCore.QPoint(0, 0)

        # Connect mouse events for interactive feedback
        self.imageItem.scene().sigMouseMoved.connect(self.mouseMove)
        self.view.setMenuEnabled(False)
        
        # Set up context menu for label editing
        self._setup_context_menu()
        
        # Custom mouse drag handling to prevent right-click interference
        self.view.mouseDragEvent = self.customMouseDragEvent

    def _setup_context_menu(self):
        """Set up the context menu for label editing operations."""
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

    def customMouseDragEvent(self, ev, axis=None):
        """
        Custom mouse drag event handler to prevent right-click interference.
        
        Parameters:
        -----------
        ev : QMouseEvent
            Mouse event
        axis : optional
            Axis parameter for original method
        """
        # Get the original method
        original_method = pg.ViewBox.mouseDragEvent.__get__(
            self.view, pg.ViewBox)

        # Call the original method only for non-right button events
        if ev.button() != QtCore.Qt.RightButton:
            original_method(ev, axis)
        else:
            ev.accept()  # Accept right button events but don't process them

    def rescaleLabels(self):
        """
        Rescale labels to ensure contiguous numbering from 0 to max_label.
        
        This method reassigns label values to eliminate gaps in the numbering
        sequence, which can occur after deleting labels.
        """
        # Create new contiguous label sequence
        newlabels = np.arange(len(np.unique(self.currentover.image)))
        oldlabels = np.unique(self.currentover.image)
        
        # Remap old labels to new labels
        for i in range(newlabels.shape[0]):
            self.currentover.image[self.currentover.image == oldlabels[i]] = newlabels[i]
            
        # Emit signal for external label tracking
        self.sigRemapLabels.emit(oldlabels, newlabels, self.currentIndex)
        self.updateImage()

    def mouseMove(self, abs_pos):
        """
        Handle mouse movement for tooltip display.
        
        Parameters:
        -----------
        abs_pos : QPointF
            Absolute position of mouse in scene coordinates
        """
        # Convert scene coordinates to image coordinates
        pos = self.imageItem.mapFromScene(abs_pos)
        row, col = int(pos.y()), int(pos.x())

        # Display tooltip with pixel information if cursor is over image
        if ((0 <= row < self.currentover.image.shape[0]) and 
            (0 <= col < self.currentover.image.shape[1])):
            message = f'X = {row};\nY = {col};\n' + \
                f'Value = {self.currentover.image[row,col]};'
            self.setToolTip(message)
        else:
            self.setToolTip('')
            
        # Update cursor position tracking
        global_pos = QtCore.QPoint(int(abs_pos.x()), int(abs_pos.y()))
        self.global_pos = self.mapToGlobal(global_pos)
        self.cursor_position = pos

    def setOverlay(self, overlay):
        """
        Set or update the overlay mask data.
        
        Parameters:
        -----------
        overlay : np.ndarray
            3D array of overlay masks (time, height, width)
        """
        if self.has_overlay:
            # Update existing overlay
            self.overlay = overlay.copy()
            self.currentover.updateImage(self.overlay[self.currentIndex, :, :])
            self.updateImage()
        else:
            # Create new overlay
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
        """
        Update the image display and overlay.
        
        Parameters:
        -----------
        autoHistogramRange : bool
            Whether to automatically adjust histogram range
        """
        super().updateImage()
        if self.has_overlay:
            # Update overlay image and colormap
            self.currentover.updateImage(
                self.overlay[self.currentIndex, :, :], autoHistogramRange=True)
            cmap = pg.colormap.get('viridis')
            self.currentover.setColorMap(cmap)
            self.currentover.setLevels((0, np.max(self.currentover.image)))
            
            # Emit signal with current number of unique labels
            newcol = str(len(np.unique(self.currentover.image.flatten())))
            self.sigChangedDefaultCol.emit(newcol)

    def contextMenuEvent(self, event):
        """
        Handle context menu events for label editing.
        
        Parameters:
        -----------
        event : QContextMenuEvent
            Context menu event
        """
        # Store click position for label operations
        point = self.imageItem.mapFromScene(event.pos())
        self.global_position = QtCore.QPoint(int(point.x()), int(point.y()))

        # Show context menu
        context_position = event.globalPos()
        self.menu.popup(context_position)
        self.resetMouseState()
        print('Ended execution of menu')

    def resetMouseState(self):
        """Reset mouse state to prevent stuck mouse events."""
        self.view.mouseIsPressed = False

        # Reset the state of other related objects
        if hasattr(self.view, 'scene'):
            scene = self.view.scene()
            if hasattr(scene, 'clickEvents'):
                scene.clickEvents = []

        # Force an update
        self.view.update()

    def donoth(self):
        """Do nothing action for context menu."""
        print('do nothing')

    def deleteLabel(self):
        """
        Delete the label at the current cursor position.
        
        This method sets all pixels with the same label value to 0 (background)
        and rescales the remaining labels to maintain contiguous numbering.
        """
        pos = self.global_position
        row, col = int(pos.y()), int(pos.x())

        # Check if position is within image bounds
        if ((0 <= row < self.currentover.image.shape[0]) and 
            (0 <= col < self.currentover.image.shape[1])):
            val = self.currentover.image[row, col]
            # Set all pixels with this label to background
            self.currentover.image[self.currentover.image == val] = 0
            self.rescaleLabels()
            self.currentover.updateImage(self.currentover.image)
            self.sigLabelDeleted.emit(self.currentIndex, val)
        else:
            return
        self.updateImage()
        

    def changeLabel(self):
        """
        Change the label at the current cursor position to the brush color.
        
        This method changes all pixels with the same label value to the
        current brush color and rescales labels to maintain contiguous numbering.
        """
        pos = self.global_position
        row, col = int(pos.y()), int(pos.x())

        # Check if position is within image bounds
        if ((0 <= row < self.currentover.image.shape[0]) and 
            (0 <= col < self.currentover.image.shape[1])):
            val = self.currentover.image[row, col]
            # Change all pixels with this label to brush color
            self.currentover.image[self.currentover.image == val] = self.parent.brushcolor
            self.rescaleLabels()
            self.currentover.updateImage(self.currentover.image)
        else:
            return
        self.updateImage()

    def mouseDoubleClickEvent(self, event):
        """
        Handle double-click events for label selection.
        
        Parameters:
        -----------
        event : QMouseEvent
            Mouse double-click event
        """
        pos = self.cursor_position
        row, col = int(pos.y()), int(pos.x())

        # Check if position is within image bounds and emit signal for label selection
        if ((0 <= row < self.currentover.image.shape[0]) and 
            (0 <= col < self.currentover.image.shape[1])):
            if event.button() == QtCore.Qt.LeftButton:
                self.sigClickedImage.emit(
                    self.currentIndex, self.currentover.image[row, col])