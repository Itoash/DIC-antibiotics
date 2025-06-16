#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:50:13 2025

@author: victorionescu
"""
    
    
def runApp():
    import sys
    from PyQt5.QtWidgets import QApplication
    from AnalysisGUI.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName('AC analysis')
    imageViewer = MainWindow()
    imageViewer.resize(1600, 900)
    imageViewer.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # If run directly, execute the main function
    runApp()