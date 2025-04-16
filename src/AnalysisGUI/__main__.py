#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:50:13 2025

@author: victorionescu
"""
    
    
def runApp():
    import sys
    from PyQt5.QtWidgets import QApplication
    from AnalysisGUI.newapp import MainWindow
    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)
    app.setApplicationName('AC analysis')
    imageViewer = MainWindow()
    imageViewer.showMaximized()
    sys.exit(app.exec_())