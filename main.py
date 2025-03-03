
"""Main window-style application."""

import sys
import numpy as np
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QSlider,QHBoxLayout,QVBoxLayout,QPushButton,QMenu,QAction,QFileDialog,QGridLayout,QWidget,QGroupBox
)
from PyQt5.QtGui import QIcon,QPixmap,QImage,QPainter
from PyQt5.QtCore import Qt,QCoreApplication
import pyqtgraph as pg
import cv2
import matplotlib.pyplot as plt
from modules.AC_processing import obtainACarray
from modules.omni_segmentation import segmentDComni



class ImageWindow(pg.ImageView):
    def __init__(self,parentwindow,name):
        super().__init__(parent = parentwindow,name = name)
        
        # Use ScatterPlotItem to draw points
        self.scatterItem = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(255, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 255, 255)
        )
        self.scatterItem.setZValue(2) # Ensure scatterPlotItem is always at top
        self.points = [] # Record Points

        self.addItem(self.scatterItem)
        
        self.scatterItem.show()
        def mousePressEvent(self, event):
            print("mouse was pressed")
            point = self.vb.mapSceneToView(event.pos()) # get the point clicked
            # Get pixel position of the mouse click
            x, y = int(point.x()), int(point.y())
            
            self.points.append([x, y])
            self.scatterItem.setPoints(pos=self.points)
            super().mousePressEvent(event)
      
class ImageStack(QWidget):
    def __init__(self,parent,stack,name):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout()
        height = stack.shape[1]
        width = stack.shape[2]
        self.images = [QPixmap.fromImage(QImage(stack[i,:,:],height,width,QImage.Format_Grayscale16)) for i in range(len(stack[:,0,0]))]
        self.labels = [QLabel().setPixmap(el) for el in self.images]
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.labels[0])
        self.setLayout(self.layout)


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__(parent=None)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setWindowTitle("Main")
        self.layout = QHBoxLayout()
        self.box = QGroupBox()
        # placeholder = QLabel("Images go here...")
        # self.layout.addWidget(placeholder)
        # self.box.setLayout(self.layout)
        
        self.image = ImageStack(self,np.full((3,100,100),0),'Image')
        self.layout.addChildWidget(self.image)
        self.box.setLayout(self.layout)
        self.setCentralWidget(self.box)
        self._createToolBar()
        self._createMenuBar()
        
        
        
    def mouseClickedEvent(self, event):
        print("mouse clicked")
        pos = event.scenePos()
        if (self.sceneBoundingRect().contains(pos)):
            mousePoint = self.plotItem.vb.mapSceneToView(pos)

            # add point to scatter item
            self.scatterItem.addPoints([mousePoint.x()], [mousePoint.y()])
        
        
        
    def closeEvent(self,event):
        QApplication.closeAllWindows
        
    def openimage(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Image")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            
            
              
            image = []
            for i in range(len(selected_files)):
                img = cv2.imread(selected_files[i],-1)
                image.append(img)
                
            image = np.asarray(image)
            self.DCview =  ImageStack(self,image,'DC')
            self.updateCentralWidget([self.DCview])
            # self.DCview.setImage(image,autoRange=True,autoLevels = False,levels=(0,65535))
            
    def updateCentralWidget(self,items):
        self.layout = QHBoxLayout()
        self.box = QGroupBox()
        for it in items:
           self.layout.addWidget(it)
         
        self.box.setLayout(self.layout)
        self.setCentralWidget(self.box)
    
    def openraw(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Directories")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            self.DCview = ImageWindow(self,'DC')
            self.ACview = ImageWindow(self,'AC')
            self.updateCentralWidget([self.DCview,self.ACview])
            dirpath = file_dialog.selectedFiles()
            dirpath = dirpath[0]
            filenames = [dirpath+"/"+f for f in os.listdir(dirpath) if f !=".DS_Store" ]
            sorted_filenames = sorted(filenames, key=lambda x: int(x[-8:-4]))
            ACstack = []
            DCstack = []
            for f in sorted_filenames:
                AC,DC = obtainACarray(f)
                AC = AC.astype(np.uint16)
                DC = DC.astype(np.uint16)
                ACstack.append(AC)
                DCstack.append(DC)
                
            self.DCstack = DCstack
            self.ACstack = ACstack
            ACstack = np.asarray(ACstack)
            DCstack = np.asarray(DCstack)
            
            self.DCview.setImage(DCstack,autoRange=True,autoLevels = True)
            self.ACview.setImage(ACstack,autoRange=True,autoLevels = False,levels=(0,65535))
            
    def segmentimage(self):
        if 'DCstack' not in self.__dict__.keys():
            
            print("Wrong!!")
            return 0
        self.segview = ImageWindow(self,'segmented')
        self.segview.show()
        segimages = segmentDComni(self.DCstack)
        self.segstack = segimages
        segimages = np.asarray(segimages)
        self.segview.setImage(segimages,autoRange=True,autoLevels=True)
        self.segview.autoLevels()
        
        
    def _createMenuBar(self):
         menuBar = self.menuBar()
         menuBar.setNativeMenuBar(True)
         # Creating menus using a QMenu object
         fileMenu = QMenu("&File", self)
         openmenu = QMenu("&Open ...",self)
         open_images = QAction("Open Processed Images",openmenu)
         open_raw = QAction("Open Raw Data",openmenu)
         
         open_images.triggered.connect(self.openimage)
         open_raw.triggered.connect(self.openraw)
         openmenu.addAction(open_images)
         openmenu.addAction(open_raw)
         fileMenu.addMenu(openmenu)
         menuBar.addMenu(fileMenu)
         # Creating menus using a title
         editMenu = menuBar.addMenu("&Edit")
         segmentbutton = QAction("Segment Image",self)
         segmentbutton.triggered.connect(self.segmentimage)
         editMenu.addAction(segmentbutton)
         menuBar.addMenu(editMenu)
         helpMenu = menuBar.addMenu("&Help")
         menuBar.addMenu(helpMenu)
    def g(self,event):
        
        modifiers = event.keyboardModifiers() & QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            print("Shift+Click")
        
        
    def _createToolBar(self):
        tools = QToolBar()
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(QApplication.closeAllWindows)
        tools.addWidget(qbtn)
        self.addToolBar(tools)
        
    def _createStatusBar(self):
        status = QStatusBar()
        status.showMessage("Normal")
        self.setStatusBar(status)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    