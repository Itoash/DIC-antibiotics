import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from copy import deepcopy
import cv2
import time as tm
from AnalysisGUI.cell_viewer import CellViewer
from AnalysisGUI.timeseries_viewer import DataVisualizerApp
from AnalysisGUI.utils.TrackerModule import obtain_network, stabilize_images
from AnalysisGUI.utils.cellprocessor import process_cells
from AnalysisGUI.widgets import OverlayImage, Graph

# Set PyQtGraph configuration
pg.setConfigOption('imageAxisOrder', 'row-major')

class TrackWindow(QtWidgets.QMainWindow):
    """
    Main window for cell tracking and visualization.
    
    This class provides a UI for viewing segmented cell images, tracking cells across frames,
    and analyzing cell lineages.
    """
    sigKeyPressed = QtCore.Signal(object)  # Signal for key press events

    def __init__(self, segBuffer, analysisBuffer, parent = None,stabilize=False):
        """
        Initialize the TrackWindow with image buffers and analysis data.
        
        Parameters:
        -----------
        segBuffer : Buffer
            Contains segmentation masks and images
        analysisBuffer : Buffer
            Contains analysis data (ACs, DCs)
        stabilize : bool, optional
            Whether to stabilize the images (reduce movement between frames)
        """
        super().__init__(parent=parent)
        self.segBuffer = segBuffer
        self.analysisBuffer = analysisBuffer
        self.brushsize = 3
        self.brushcolor = 0
        self.highlight = None  # For highlighting selected cells
        
        # Stabilize images if requested
        if stabilize:
            self._stabilize_images()
            
        # Set up the main UI components
        self._setup_ui()
        
        # Set up image network and graph visualization
        self._setup_network()
        
        # Connect signals and custom event handlers
        self._connect_signals()
        
        # Display the window
        self.showMaximized()

    #################################
    # Initialization Helper Methods #
    #################################
    
    def _stabilize_images(self):
        """Stabilize images to reduce movement between frames."""
        if len(self.segBuffer.images) == len(self.analysisBuffer.ACs):
            print(f"Analysis buffer is in sync with segBuffer: {len(self.segBuffer.images)}")
            newDC, segs, newAC = stabilize_images(
                self.segBuffer.images, self.segBuffer.masks, self.analysisBuffer.ACs)
            segs = [s.astype(int) for s in segs]
            
            self.segBuffer.masks = segs
            for DC in newDC:
                minval = np.mean(DC[DC.astype(float) != float(0)])
                DC[DC.astype(float) == float(0)] = minval
            self.segBuffer.images = newDC
            self.analysisBuffer.DCs = deepcopy(newDC)
            self.analysisBuffer.ACs = newAC
        else:  # segBuffer and analysisBuffer are not synced
            print(f"Analysis buffer is NOT in sync with segBuffer: {len(self.analysisBuffer.ACs),len(self.segBuffer.images)}")
            newDC, segs = stabilize_images(
                self.segBuffer.images, self.segBuffer.masks)
            
            segs = [s.astype(int) for s in segs]
            self.segBuffer.masks = segs
            for DC in newDC:
                minval = np.mean(DC[DC.astype(float) != float(0)])
                DC[DC.astype(float) == float(0)] = minval
            self.segBuffer.images = newDC

    def _setup_ui(self):
        """Set up the main UI components and layout."""
        # Create main widget and layout
        self.wid = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self.wid)
        
        # Create a vertical splitter to contain the image viewer and graph
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(splitter)
        
        # Create the toolbar
        self._setup_toolbar()
        
        # Create image display widget
        self.DCplot = OverlayImage(self)
        
        # Create graph plot for cell lineage visualization
        self.GraphPlot = pg.PlotWidget()
        self.Graph = Graph()
        self.GraphPlot.addItem(self.Graph)
        
        # Add widgets to splitter
        splitter.addWidget(self.DCplot)
        splitter.addWidget(self.GraphPlot)
        
        # Set the image and overlay
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                          't': 0, 'x': 2, 'y': 1, 'c': None})
        self.DCplot.setOverlay(np.asarray(self.segBuffer.masks))
        
        # Hide unnecessary UI elements
        self.DCplot.ui.roiBtn.hide()
        self.DCplot.ui.menuBtn.hide()
        self.DCplot.view.setMenuEnabled(False)
        
        # Set layout and central widget
        self.layout.setMenuBar(self.toolbar)
        self.wid.setLayout(self.layout)
        self.setCentralWidget(self.wid)
        
        # Auto-range the image display
        self.DCplot.autoRange()
        print(f'minvalue in DC: {np.min(self.DCplot.imageItem.image)}')

    def _setup_toolbar(self):
        """Set up the toolbar with all its buttons and controls."""
        self.toolbar = QtWidgets.QToolBar()
        
        # Drawing tools
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

        # Opacity slider
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
        
        self.remakeNet = QtWidgets.QAction("Recompute network", self)
        self.remakeNet.triggered.connect(self.recomputeNet)
        self.toolbar.addAction(self.remakeNet)
        
        self.deleteImg = QtWidgets.QAction("Delete current image", self)
        self.deleteImg.triggered.connect(self.deleteImage)
        self.toolbar.addAction(self.deleteImg)
        
        self.stabilizeImages = QtWidgets.QAction("Stabilize image stack (permanent)", self)
        self.stabilizeImages.triggered.connect(self._stabilize_images)
        self.toolbar.addAction(self.stabilizeImages)
        # Visualization tools
        self.openTS = QtWidgets.QAction("Open Cell Time Series", self)
        self.openTS.triggered.connect(self.openTimeSeries)
        self.toolbar.addAction(self.openTS)
        
        self.openVis = QtWidgets.QAction("Open Cell Visualiser", self)
        self.openVis.triggered.connect(self.openVisual)
        self.toolbar.addAction(self.openVis)
        
        # Cell filtering tools
        self.filter = QtWidgets.QAction("Filter Cells", self)
        self.filter.setCheckable(False)
        self.filter.triggered.connect(self.filterNodes)
        self.toolbar.addAction(self.filter)

        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(["Convexity", "Size", "Aspect Ratio"])
        self.combobox.currentIndexChanged.connect(self.changeText)
        self.toolbar.addWidget(self.combobox)

        self.convIn = QtWidgets.QLineEdit(str(0.8))
        self.convIn.setMaxLength(4)
        self.convIn.setMaximumWidth(50)
        self.toolbar.addWidget(self.convIn)

    def _setup_network(self):
        """Set up the image network and graph visualization."""
        # Create network from segmentation masks
        self.imagenet = obtain_network(self.segBuffer.masks.copy())
        
        # Handle error case where network creation fails
        if isinstance(self.imagenet, int):
            idx = self.imagenet
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(
                f"Empty segmentation image contained at index {idx} in file {self.analysisBuffer.names[idx]}")
            msg.setInformativeText(
                'Retry tracking with valid segmentations!')
            msg.setWindowTitle("TypeError")
            msg.exec_()
            self.close()
            return
        
        # Store network in segBuffer
        self.segBuffer.imagenet = deepcopy(self.imagenet)
        
        # Get graph data and set it to the Graph widget
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)

    def _connect_signals(self):
        """Connect signals and custom event handlers."""
        # Connect signals between widgets
        self.DCplot.sigChangedDefaultCol.connect(self.changeCol)
        self.DCplot.sigClickedImage.connect(self.Graph.cellClicked)
        self.DCplot.sigTimeChanged.connect(self.switchoffHighlight)
        
        self.Graph.sigPointSelected.connect(self.highlightCell)
        self.Graph.sigPanView.connect(self.panView)
        self.Graph.sigPointDeselected.connect(self.switchoffHighlight)
        self.Graph.sigNodesLinked.connect(self.linkNodes)
        self.Graph.sigNodeDisconnected.connect(self.unlinkNode)
        self.Graph.sigDeletedLineage.connect(self.deleteLineage)
        self.Graph.sigContextMenuOpened.connect(self.supersedeRClick)
        
        # Custom event handlers for key press and mouse drag
        self.GraphPlot.keyPressEvent = self.customKeyPress
        self.sigKeyPressed.connect(self.Graph.keyPressEvent)
        self.GraphPlot.plotItem.vb.mouseDragEvent = self.customMouseDragEvent

    ###########################
    # Event Handling Methods  #
    ###########################
    
    def customKeyPress(self, ev):
        """Custom key press event handler."""
        # Get the original method
        original_method = pg.PlotWidget.keyPressEvent.__get__(
            self.GraphPlot, pg.PlotWidget)

        # Emit signal for key press
        self.sigKeyPressed.emit(ev)
        
        # Call the original method
        original_method(ev)

    def customMouseDragEvent(self, ev, axis=None):
        """Custom mouse drag event handler."""
        # Get the original method
        original_method = pg.ViewBox.mouseDragEvent.__get__(
            self.GraphPlot.plotItem.vb, pg.ViewBox)
            
        # Call the original method only for non-right button events
        if ev.button() != QtCore.Qt.RightButton:
            original_method(ev, axis)
        else:
            ev.accept()  # Accept right button events but don't process them
    
    def supersedeRClick(self):
        """Handle right click in graph by clearing mouse state."""
        self.GraphPlot.clearMouse()
    
    def panView(self, x, y):
        """Pan the graph view by the specified amount."""
        self.GraphPlot.getViewBox().translateBy(x=x, y=y)

    ###########################
    # Cell Selection Methods  #
    ###########################
    
    def switchoffHighlight(self, t):
        """Remove cell highlight when selection changes."""
        if self.highlight is not None:
            print("Clear highlight")
            self.highlight.clear()
            self.highlight.update()
            self.DCplot.update()

    def highlightCell(self, t, label):
        """Highlight a cell in the image view."""
        # Set to the correct time frame
        self.DCplot.setCurrentIndex(int(t))
        
        # Create a mask for the selected cell
        outlineimg = self.DCplot.currentover.image.copy()
        outlineimg[outlineimg != label] = 0
        outlineimg = outlineimg/label
        
        # Find contour of the cell
        contour, _ = cv2.findContours(outlineimg.astype(
            'uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Extract contour points
        contourx = contour[0][:, 0, 0]
        contoury = contour[0][:, 0, 1]
        contourx = np.append(contourx, contourx[0])  # Close the contour
        contoury = np.append(contoury, contoury[0])
        
        # Create a red pen for drawing
        pen = pg.mkPen('r', width=5)
        
        # Create or update the highlight
        if self.highlight is None:
            self.highlight = pg.PlotCurveItem(x=contourx, y=contoury, pen=pen)
            self.DCplot.addItem(self.highlight)
            self.highlight.setZValue(20)
        else:
            self.highlight.updateData(x=contourx, y=contoury, pen=pen)
            self.highlight.setZValue(20)

    #############################
    # Network Management Methods #
    #############################
    
    def linkNodes(self, current, target):
        """Link two nodes in the cell lineage graph."""
        # Don't link nodes in the same time frame
        if current[0] == target[0]:
            return
            
        # Link from earlier to later time point
        if current[0] < target[0]:
            self.imagenet.linknodes(current, target)
        else:
            self.imagenet.linknodes(target, current)
            
        # Update the graph
        self.imagenet.makeTree()
        self.imagenet.virtualizeCoords()
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)
        self.Graph.updateGraph()

    def unlinkNode(self, nodepos, direction):
        """Unlink a node in the cell lineage graph."""
        self.imagenet.unlinknode(nodepos, direction)
        
        # Update the graph
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)
        self.Graph.updateGraph()
    
    def deleteLineage(self, nodename):
        """Delete a lineage from the cell lineage graph."""
        currentIdx = self.DCplot.currentIndex
        self.switchoffHighlight(0)
        self.Graph.selectedIndices = None
        overlays = self.DCplot.overlay
        remove_list = self.imagenet.filterByLineage(
                    nodename)
        tic = tm.time()
        for node in remove_list:
            k = overlays[int(node[0]), :, :]
            k[k == float(node[1])] = 0
            overlays[int(node[0]), :, :] = k
        print(f'Removing nodes from array took {round(tm.time()-tic,3)}s')
        
        # Rescale remaining labels
        for i in range(overlays.shape[0]):
            self.DCplot.setCurrentIndex(i)
            self.DCplot.rescaleLabels()
        
        # Update display and network
        
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        self.DCplot.setOverlay(overlays)
        self.DCplot.updateImage()
        self.DCplot.setCurrentIndex(currentIdx)
        # Recompute network
        self.imagenet = obtain_network(
            [overlays[i, :, :] for i in range(overlays.shape[0])])
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)
        self.switchoffHighlight(0)
        

    def rescaledLabels(self, oldlabels, newlabels, currIdx):
        """Update the network after rescaling labels."""
        self.imagenet.reassignLabels(oldlabels, newlabels, currIdx)
        
        # Update the graph
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)
        self.switchoffHighlight(0)
        
    def recomputeNet(self):
        """Recompute the cell network from current overlay images."""
        imgs = self.DCplot.overlay
        imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        
        # Create new network
        self.imagenet = obtain_network(imgs)
        
        # Update the graph
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)
    
    def filterNodes(self):
        """Filter cells based on selected criteria."""
        currentIdx = self.DCplot.currentIndex
        overlays = self.DCplot.overlay
        choice = self.combobox.currentIndex()
        
        # Apply filter based on selected criteria
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
        
        # Remove filtered nodes from overlays
        tic = tm.time()
        for node in remove_list:
            k = overlays[int(node[0]), :, :]
            k[k == float(node[1])] = 0
            overlays[int(node[0]), :, :] = k
        print(f'Removing nodes from array took {round(tm.time()-tic,3)}s')
        
        # Rescale remaining labels
        for i in range(overlays.shape[0]):
            self.DCplot.setCurrentIndex(i)
            self.DCplot.rescaleLabels()
        
        # Update display and network
        self.DCplot.setCurrentIndex(0)
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        self.DCplot.setOverlay(overlays)
        self.DCplot.updateImage()
        self.DCplot.setCurrentIndex(currentIdx)
        # Recompute network
        self.imagenet = obtain_network(
            [overlays[i, :, :] for i in range(overlays.shape[0])])
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)
        self.switchoffHighlight(0)

    #############################
    # Image Management Methods  #
    #############################
    
    def resetImages(self):
        """Reset images to their original state from segBuffer."""
        currentIdx = self.DCplot.currentIndex
        
        # Reset the image and overlay
        self.DCplot.setImage(np.asarray(self.segBuffer.images), axes={
                             't': 0, 'x': 2, 'y': 1, 'c': None})
        img = np.asarray(self.segBuffer.masks)
        self.DCplot.setOverlay(img)
        
        # Reset the network
        self.imagenet = deepcopy(self.segBuffer.imagenet)
        graphnodes, graphedges, text, meta = self.imagenet.exportGraph()
        self.Graph.setData(pos=graphnodes, adj=graphedges,
                        text=text, size=15, meta=meta)
                        
        # Set index to current or last valid frame
        self.DCplot.setCurrentIndex(min(currentIdx, img.shape[0]-1))
    
    def checkpoint(self):
        """Save current state to segBuffer."""
        imgs = self.DCplot.overlay
        imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        self.segBuffer.masks = deepcopy(imgs)
        self.segBuffer.imagenet = deepcopy(self.imagenet)
    
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
            
        # Update the network and images
        self.resetImages()
        self.recomputeNet()
        self.switchoffHighlight(0)

    #########################
    # Drawing Tool Methods  #
    #########################
    
    def changeSize(self, text):
        """Change the brush size for drawing."""
        try:
            newsize = int(text)
            self.brushsize = newsize
            
            # Update the drawing kernel if in draw mode
            if self.drawButton.isChecked():
                val = self.brushcolor
                kern = np.full((self.brushsize, self.brushsize), val)
                cen = self.brushsize % 2
                self.DCplot.currentover.setDrawKernel(
                    kern, mask=None, center=(cen, cen), mode='set')
        except:
            pass

    def changeCol(self, text):
        """Change the brush color for drawing."""
        try:
            newcol = int(text)
            self.brushcolor = newcol
            self.bColIn.setText(str(self.brushcolor))
            
            # Update the drawing kernel if in draw mode
            if self.drawButton.isChecked():
                val = self.brushcolor
                kern = np.full((self.brushsize, self.brushsize), val)
                cen = self.brushsize % 2
                self.DCplot.currentover.setDrawKernel(
                    kern, mask=None, center=(cen, cen), mode='set')
        except:
            pass

    def drawMode(self, checked):
        """Toggle drawing mode on/off."""
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
        """Change the opacity of the overlay."""
        self.DCplot.currentover.setOpacity(value/1000)
    
    def changeText(self, idx):
        """Update filter value text based on combobox selection."""
        match idx:
            case 0:  # Convexity
                text = str(0.8)
            case 1:  # Size
                text = str(100)
            case 2:  # Aspect Ratio
                text = str(1.1)
        self.convIn.setText(text)

    ##########################
    # Visualization Methods  #
    ##########################

    def openVisual(self):
        """Open the cell visualizer window."""
        # Only open if buffers are in sync
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            # Get lineage and data
            lineage = self.imagenet.getLineageDict()
            ACs = self.analysisBuffer.ACs[:]
            DCs = self.analysisBuffer.DCs[:]
            labels = self.segBuffer.masks[:]
            times = self.analysisBuffer.abstimes[:]
            times = list(sorted(times))
            times = np.asarray([t-times[0] for t in times]).astype(int)
            
            # Process cell data
            tic = tm.time()
            cell_data = process_cells(lineage, ACs, DCs, labels, times)
            cell_keys = list(cell_data.keys())
            cell_keys.sort()
            cell_data = {i: cell_data[i] for i in cell_keys}
            print(f'Calculating per cell took {round(tm.time()-tic,3)} for {len(cell_data.keys())} cells.')
            
            # Open the visualizer
            self.vis = CellViewer(cell_data, ACs, DCs, times,self)
            self.vis.show()
            
    def openTimeSeries(self):
        """Open the time series visualizer window."""
        # Only open if buffers are in sync
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            # Get lineage and data
            lineage = self.imagenet.getLineageDict()
            ACs = self.analysisBuffer.ACs[:]
            DCs = self.analysisBuffer.DCs[:]
            labels = self.segBuffer.masks[:]
            times = self.analysisBuffer.abstimes[:]
            times = list(sorted(times))
            times = np.asarray([t-times[0] for t in times]).astype(int)
            print(len(times))
            print(len(ACs))
            # Process cell data
            tic = tm.time()
            cell_data = process_cells(lineage, ACs, DCs, labels, times)
            cell_keys = list(cell_data.keys())
            cell_keys.sort()
            cell_data = {i: cell_data[i] for i in cell_keys}
            print(f'Calculating per cell took {round(tm.time()-tic,3)} for {len(cell_data.keys())} cells.')
            
            # Open the visualizer
            self.vis = DataVisualizerApp(cell_data,self)
            self.vis.show()

    def saveCells(self,path):
        """Save the current cell data to a file."""
        # Only save if buffers are in sync
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            lineage = self.imagenet.getLineageDict()
            ACs = self.analysisBuffer.ACs[:]
            DCs = self.analysisBuffer.DCs[:]
            labels = self.segBuffer.masks[:]
            times = self.analysisBuffer.abstimes[:]
            times = list(sorted(times))
            times = np.asarray([t-times[0] for t in times]).astype(float)
            
            # Process cell data
            tic = tm.time()
            cell_data = process_cells(lineage, ACs, DCs, labels, times)
            cell_keys = list(cell_data.keys())
            cell_keys.sort()
            cell_data = {i: cell_data[i] for i in cell_keys}
            print(f'Calculating per cell took {round(tm.time()-tic,3)} for {len(cell_data.keys())} cells.')
            writeCellDict(cell_data, path)
            cv2.imwritemulti(path+'/segmentation.tif', self.segBuffer.masks)
            cv2.imwritemulti(path+'/DCs.tif', self.segBuffer.images)
            cv2.imwritemulti(path+'/ACs.tif', self.analysisBuffer.ACs)
        else:
            print('Cannot save cells, buffers are not in sync.')

def writeCellDict(cell_data, path):
    """Write cell data to a file."""
    import csv
    import os
    name = "cell_data.csv"
    fullpath = os.path.join(path, name)
    with open(fullpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in cell_data.items():
            writer.writerow([key])
            for k,v in value.items():
                if 'Interior contour' not in k and  'Total contour' not in k:
                    if isinstance(v, list):
                        writer.writerow([k] + v)
                    elif isinstance(v, np.ndarray):
                        writer.writerow([k] + v.tolist())
            writer.writerow([])  # Empty line between cells
    print(f'Cell data saved to {fullpath}')
    return fullpath

