#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:00:54 2025

@author: victorionescu
"""

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
import PyQt5
import pyqtgraph as pg
import cv2
import matplotlib.pyplot as plt
from modules.AC_processing import obtainACarray
from modules.omni_segmentation import segmentDComni
from PIL import Image
class ImageStack(QWidget):
    def __init__(self,parent,stack,name):
        super().__init__(parent=parent)
        self.images = stack.astype('uint16')
        self.layout = QVBoxLayout
        self.slider = QSlider()
        self.position = 0
        self.displayedImage = QLabel 
        self.slider = QSlider()
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.images[:,0,0])-1)
        self.slider.valueChanged.connect(self.scrollImage)
        self.slider.setOrientation(Qt.Horizontal)
        self.updateFrame()
        self.setLayout(self.layout)
    def updateFrame(self):
        
        height, width = self.images[0,:,:].shape
        bytesPerLine = 2 * width
        self.currentimage = QImage(self.images[self.position,:,:].data, width, height, bytesPerLine, QImage.Format_Grayscale16)
        self.displayedImage.setPixmap(QPixmap.fromImage(self.currentimage))
        
        
    def scrollImage(self):
        selectedframe = self.slider.value()
        self.position = selectedframe
        self.updateFrame()
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
        dummy = np.full((3,1000,1000),0)
        dummy[1,:,:] = 30000
        dummy[2,:,:] = 65535
        self.image = ImageStack(self,np.full((3,1000,1000),0),'Image')
        self.image.show()
        self.layout.addChildWidget(self.image)
        self.box.setLayout(self.layout)
        self.setCentralWidget(self.image)
        self._createToolBar()
        self._createMenuBar()
        
        
        
    def mouseClickedEvent(self, event):
        print("mouse clicked")
    
        
        
        
        
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
    
    
            
   
        self.segview.autoLevels()
        
        
    def _createMenuBar(self):
         menuBar = self.menuBar()
         menuBar.setNativeMenuBar(True)
         # Creating menus using a QMenu object
         fileMenu = QMenu("&File", self)
         openmenu = QMenu("&Open ...",self)
         open_images = QAction("Open Processed Images",openmenu)
        
         
         open_images.triggered.connect(self.openimage)
        
         openmenu.addAction(open_images)
      
         fileMenu.addMenu(openmenu)
         menuBar.addMenu(fileMenu)
         # Creating menus using a title
         editMenu = menuBar.addMenu("&Edit")
        
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