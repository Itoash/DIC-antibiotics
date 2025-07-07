import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeWidget, QTreeWidgetItem, QSplitter,
                            QPushButton, QLabel, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5 import QtCore

class DataVisualizerApp(QMainWindow):
    def __init__(self, data_dict,parent):
        super().__init__(parent=parent)
        self.data_dict = data_dict
        self.show_all_objects = False
        self.show_lineage = False
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Time Series Data Visualizer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Create a splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left side: Tree view of data and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Tree widget for data selection
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(['Objects/Attributes'])
        self.populate_tree()
        self.update_timer = QtCore.QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)
        self.tree_widget.itemSelectionChanged.connect(lambda: self.update_timer.start(100))
        left_layout.addWidget(self.tree_widget)
        
        # Add multi-object selection checkbox
        self.multi_obj_checkbox = QCheckBox("Show selected parameter for all objects")
        self.multi_obj_checkbox.stateChanged.connect(self.toggle_multi_object_view)
        left_layout.addWidget(self.multi_obj_checkbox)
        
        # Add lineage selection checkbox
        self.lineage_checkbox = QCheckBox("Show selected parameter for all objects in current lineage")
        self.lineage_checkbox.stateChanged.connect(self.toggle_lineage_view)
        left_layout.addWidget(self.lineage_checkbox)
        
        # Add range control buttons
        range_box = QGroupBox("Range Controls")
        range_layout = QVBoxLayout(range_box)
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_range)
        range_layout.addWidget(reset_btn)
        
        self.range_info = QLabel("No range selected")
        range_layout.addWidget(self.range_info)
        
        left_layout.addWidget(range_box)
        
        splitter.addWidget(left_widget)
        
        # Right side: Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Create main plot widget
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('bottom', 'Time')
        self.plot.setLabel('left', 'Value')
        self.plot.plotItem.vb.enableAutoRange()
        plot_layout.addWidget(self.plot)
        
        # Create region selection plot widget
        self.region_plot = pg.PlotWidget(height=150)
        
        self.region_plot.showGrid(x=True, y=True)
        self.region_plot.setLabel('bottom', 'Time')
        
        # Create a LinearRegionItem for range selection
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self.update_plot_range)
        self.region_plot.addItem(self.region)
        
        plot_layout.addWidget(self.region_plot)
        
        splitter.addWidget(plot_widget)
        splitter.setSizes([300, 900])  # Initial sizes of the two sections
        
        self.setCentralWidget(main_widget)
        
        # Store current plot items so we can update both plots
        self.plotted_items = []
        self.overview_items = []
        
    def populate_tree(self):
        """Populate the tree widget with data from the nested dict"""
        self.tree_widget.clear()
        
        # Add objects as top-level items
        for obj_name, obj_data in self.data_dict.items():
            obj_item = QTreeWidgetItem([obj_name])
            self.tree_widget.addTopLevelItem(obj_item)
            
            # Add attributes as children
            for attr_name in obj_data.keys():
                if attr_name != 'times' and attr_name != 'position' and attr_name != 'Interior contour' and attr_name.lower() != 'total contour':  # Assuming 'time' might be a special key
                    attr_item = QTreeWidgetItem([attr_name])
                    obj_item.addChild(attr_item)
        
        self.tree_widget.expandAll()
    
    def toggle_multi_object_view(self):
        """Toggle between normal view and multi-object parameter view"""
        self.show_all_objects = self.multi_obj_checkbox.isChecked()
        self.update_plot()
    def toggle_lineage_view(self):
        self.show_lineage = self.lineage_checkbox.isChecked()
        self.update_plot()
    def update_plot(self):
        """Update the plot based on selected items"""
        self.plot.clear()
        self.region_plot.clear()
        self.region_plot.addItem(self.region)
        
        # Clear stored items
        self.plotted_items = []
        self.overview_items = []
        
        selected_items = self.tree_widget.selectedItems()
        
        if not selected_items:
            return
        
        # Set different colors for each plot line
        # colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        # color_idx = 0
        keys = list(self.data_dict.keys())
        keys = [str(k.split('_')[1]) for k in keys]
        keys = list(set(keys))

        colors = pg.colormap.get('rainbow',source='matplotlib')
        colormap = {k:colors[i/(len(keys))] for i,k in enumerate(keys)}
        color_idx = 0
        
        # Track min/max time values to set region boundaries
        min_time = float('inf')
        max_time = float('-inf')
        
        if self.show_all_objects or self.show_lineage:
            # Get selected parameters (attributes)
            selected_params = []
            for item in selected_items:
                if item.parent():  # This is an attribute
                    param_name = item.text(0)
                    if param_name not in selected_params:
                        selected_params.append(param_name)
            
            if self.show_lineage:
                selected_cells = []
                for item in selected_items:
                    if item.parent():  # This is an attribute, use parent name
                        param_name = item.parent().text(0)
                        if param_name not in selected_params:
                            selected_cells.append(param_name)
                    else:
                        param_name = item.text(0) # this is a cell, use its name
                        if param_name not in selected_params:
                            selected_cells.append(param_name)
                objects = [item for item in self.data_dict.items() if any(sub in item[0] for sub in selected_cells)]
            else:
                objects = self.data_dict.items()
            # For each selected parameter, plot it for all objects that have it
            for param_name in selected_params:
                for obj_name, obj_data in objects:
                    if param_name in obj_data:
                        # Get time values (x-axis)
                        time_values = self.data_dict[obj_name].get('times', list(range(len(self.data_dict[obj_name][param_name]))))
                        
                        # Update min/max time
                        min_time = min(min_time, min(time_values))
                        max_time = max(max_time, max(time_values))
                        
                        # Get attribute values (y-axis)
                        attr_values = self.data_dict[obj_name][param_name]
                        # Create plot with a unique name (object_attribute)
                        pen = pg.mkPen(color=colormap[obj_name.split('_')[1]], width=2)
                        
                        # Add to main plot
                        plot_item = self.plot.plot(time_values, attr_values, name=f"{obj_name}_{param_name}", 
                                                pen=pen, symbol='o', symbolSize=5)
                        self.plotted_items.append((plot_item, time_values, attr_values))
                        
                        # Add to overview plot (region selection)
                        overview_item = self.region_plot.plot(time_values, attr_values, pen=pen)
                        self.overview_items.append(overview_item)
                        
                        color_idx += 1



        else:
            # Original behavior - plot only selected items
            for item in selected_items:
                # Check if this is an attribute item (has a parent)
                if item.parent():
                    obj_name = item.parent().text(0)
                    attr_name = item.text(0)
                    
                    # Get time values (x-axis)
                    time_values = self.data_dict[obj_name].get('times', list(range(len(self.data_dict[obj_name][attr_name]))))
                    
                    # Update min/max time
                    min_time = min(min_time, min(time_values))
                    max_time = max(max_time, max(time_values))
                    
                    # Get attribute values (y-axis)
                    attr_values = self.data_dict[obj_name][attr_name]
                    
                    # Create plot with a unique name (object_attribute)
                    pen = pg.mkPen(color=colors[obj_name.split('_')[1]], width=2)
                    
                    # Add to main plot
                    plot_item = self.plot.plot(time_values, attr_values, name=f"{obj_name}_{attr_name}", 
                                            pen=pen, symbol='o', symbolSize=5)
                    self.plotted_items.append((plot_item, time_values, attr_values))
                    
                    # Add to overview plot (region selection)
                    overview_item = self.region_plot.plot(time_values, attr_values, pen=pen)
                    self.overview_items.append(overview_item)
                    
                    color_idx += 1
        
        # Add legend
        self.plot.addLegend()
        
        # Set region boundaries if we have data
        if min_time != float('inf') and max_time != float('-inf'):
            self.region.setRegion([min_time, max_time])
    
    def update_plot_range(self):
        """Update the main plot's view range based on region selection"""
        region_min, region_max = self.region.getRegion()
        self.plot.setXRange(region_min, region_max)
        
        # Update range info label
        self.range_info.setText(f"Selected range: {region_min:.2f} to {region_max:.2f}")
    
    def reset_range(self):
        """Reset the plot view to show all data"""
        min_time = float('inf')
        max_time = float('-inf')
        
        # Find the global min/max time across all plotted data
        for _, time_values, _ in self.plotted_items:
            if len(time_values) > 0:
                min_time = min(min_time, min(time_values))
                max_time = max(max_time, max(time_values))
        
        if min_time != float('inf') and max_time != float('-inf'):
            self.region.setRegion([min_time, max_time])
        
    def update_data(self,newdata_dict):
        self.data_dict = newdata_dict
        self.populate_tree()
        self.update_plot()
        

