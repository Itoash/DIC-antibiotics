import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from copy import deepcopy
from AnalysisGUI.CellImageViewer import CellViewer
import cv2
from AnalysisGUI.utils.TrackerModule import obtain_network, stabilize_images
import time as tm
from AnalysisGUI.CellVisExample import DataVisualizerApp
from AnalysisGUI.utils.cellprocessor import process_cells
from AnalysisGUI.widgets import OverlayImage, Graph
from copy import deepcopy
pg.setConfigOption('imageAxisOrder', 'row-major')
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
                self.analysisBuffer.DCs = deepcopy(newDC)
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
        self.segBuffer.masks = deepcopy(imgs)
        self.segBuffer.imagenet = deepcopy(self.imagenet)
        
    def deleteImage(self):
        currentIdx = self.DCplot.currentIndex
        if len(self.segBuffer.images) == len(self.analysisBuffer.DCs):
            print('Deleting image...')
            self.segBuffer.images.pop(currentIdx)
            
            self.segBuffer.masks.pop(currentIdx)
            
            self.analysisBuffer.ACs.pop(currentIdx)
            self.analysisBuffer.DCs.pop(currentIdx)
            self.analysisBuffer.names.pop(currentIdx)
            
            self.analysisBuffer.times.pop(currentIdx)
            
            self.analysisBuffer.abstimes.pop(currentIdx)
            
        self.recomputeNet()
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