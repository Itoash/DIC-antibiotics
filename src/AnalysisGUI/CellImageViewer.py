import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


class CellViewer(QtWidgets.QMainWindow):
    """Main window for visualizing AC/DC image stacks with object contours."""
    
    def __init__(self, data_dict, ac_images, dc_images, globaltimes, parent=None):
        """
        Initialize the main visualization window.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary of {object_name: {property: data}} where each object has contour data
            Each object should have a 'times' list of time values indicating when it appears
        ac_images : list
            List of AC images (numpy arrays), one per timepoint
        dc_images : list
            List of DC images (numpy arrays), one per timepoint
        globaltimes: np.ndarray[float,ndim=1]
            Array of time values to set as x axis labels (in minutes)
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.data_dict = data_dict
        self.ac_images = ac_images
        self.dc_images = dc_images
        self.globalTimes = globaltimes.astype(float)
        
        # Create a mapping between time values and image indices
        self.time_to_index = {time: idx for idx, time in enumerate(self.globalTimes)}
        self.index_to_time = {idx: time for idx, time in enumerate(self.globalTimes)}
        
        self.current_time_value = self.globalTimes[0] if len(self.globalTimes) > 0 else 0
        self.current_image_index = 0
        self.contour_items = {'ac': [], 'dc': []}
        self.current_object = None
        self.time_change_lock = False  # Lock to prevent recursive time change calls
        
        self.setup_ui()
        self.populate_tree()
        self.load_images()
        
        self.setWindowTitle("Data Visualization")
        self.resize(1200, 800)
        
    def setup_ui(self):
        """Set up the user interface components."""
        # Main widget and layout
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # Splitter for tree view and images
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Tree view for object names
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabel("Objects")
        self.tree_widget.setMinimumWidth(200)
        self.tree_widget.itemClicked.connect(self.on_object_selected)
        self.main_splitter.addWidget(self.tree_widget)
        
        # Right side for images
        self.image_widget = QtWidgets.QWidget()
        self.image_layout = QtWidgets.QVBoxLayout(self.image_widget)
        self.main_splitter.addWidget(self.image_widget)
        
        # Splitter for AC and DC images
        self.image_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.image_layout.addWidget(self.image_splitter)
        
        # AC Image View
        self.ac_view_widget = QtWidgets.QWidget()
        self.ac_view_layout = QtWidgets.QVBoxLayout(self.ac_view_widget)
        self.ac_view_layout.setContentsMargins(0, 0, 0, 0)
        self.ac_label = QtWidgets.QLabel("AC Images")
        self.ac_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ac_view_layout.addWidget(self.ac_label)
        self.ac_view = pg.ImageView()
        self.ac_view_layout.addWidget(self.ac_view)
        self.image_splitter.addWidget(self.ac_view_widget)
        
        # DC Image View
        self.dc_view_widget = QtWidgets.QWidget()
        self.dc_view_layout = QtWidgets.QVBoxLayout(self.dc_view_widget)
        self.dc_view_layout.setContentsMargins(0, 0, 0, 0)
        self.dc_label = QtWidgets.QLabel("DC Images")
        self.dc_label.setAlignment(QtCore.Qt.AlignCenter)
        self.dc_view_layout.addWidget(self.dc_label)
        self.dc_view = pg.ImageView()
        self.dc_view_layout.addWidget(self.dc_view)
        self.image_splitter.addWidget(self.dc_view_widget)
        
        # Connect time change signals
        self.ac_view.sigTimeChanged.connect(self.on_ac_time_changed)
        self.dc_view.sigTimeChanged.connect(self.on_dc_time_changed)
        
        # Set splitter sizes
        self.main_splitter.setSizes([200, 1000])
        self.image_splitter.setSizes([400, 400])
        
    def populate_tree(self):
        """Populate the tree view with sorted object names."""
        # Sort object names
        sorted_objects = sorted(self.data_dict.keys())
        
        # Add each object to the tree
        for obj_name in sorted_objects:
            item = QtWidgets.QTreeWidgetItem(self.tree_widget)
            item.setText(0, obj_name)
            item.setData(0, QtCore.Qt.UserRole, obj_name)
        
    def load_images(self):
        """Load AC and DC image stacks into their respective views."""
        if self.ac_images:
            # Convert list of 2D arrays to a 3D array (time, height, width)
            ac_stack = np.array(self.ac_images)
            self.ac_view.setImage(ac_stack, xvals=self.globalTimes)
            
        if self.dc_images:
            # Convert list of 2D arrays to a 3D array (time, height, width)
            dc_stack = np.array(self.dc_images)
            self.dc_view.setImage(dc_stack, xvals=self.globalTimes)
    
    def synchronize_time(self, time_or_index, source):
        """
        Synchronize the time between AC and DC views.
        
        Parameters:
        -----------
        time_or_index : float or int
            Time value or index from the slider
        source : str
            Which view triggered the change ('ac' or 'dc')
        """
        if self.time_change_lock:
            return
            
        self.time_change_lock = True
        
        # pyqtgraph time change might give us the actual time value, not the index
        # We need to find the nearest index for this time
        if isinstance(time_or_index, float):
            # Find the nearest time value in globalTimes
            closest_idx = np.abs(self.globalTimes - time_or_index).argmin()
            self.current_image_index = closest_idx
            self.current_time_value = self.globalTimes[closest_idx]
        else:
            # We've been given an integer index directly
            self.current_image_index = int(time_or_index)
            self.current_time_value = self.globalTimes[self.current_image_index]
        
        print(f"Time synchronized: index={self.current_image_index}, time={self.current_time_value}")
        
        # Update the other view without triggering its own signal
        if source == 'ac' and self.dc_view.image is not None:
            self.dc_view.setCurrentIndex(self.current_image_index)
        elif source == 'dc' and self.ac_view.image is not None:
            self.ac_view.setCurrentIndex(self.current_image_index)
            
        # Clear existing contours first
        self.clear_contours()
        
        # Update contours for the current object
        if self.current_object:
            self.update_contours()
            
        self.time_change_lock = False
    
    def on_ac_time_changed(self, image_view, time_idx):
        """Handle time change in AC view."""
        self.synchronize_time(time_idx, 'ac')
    
    def on_dc_time_changed(self, image_view, time_idx):
        """Handle time change in DC view."""
        self.synchronize_time(time_idx, 'dc')
    
    def on_object_selected(self, item, column):
        """Handle object selection in the tree view."""
        obj_name = item.data(0, QtCore.Qt.UserRole)
        self.current_object = obj_name
        
        # Clear existing contours
        self.clear_contours()
        
        # Get object data
        obj_data = self.data_dict.get(self.current_object, {})
        if not obj_data:
            return
            
        # Get the list of time values where this object appears
        object_times = obj_data.get('times', [])
        if not len(object_times):
            return
            
        # Find the first time when the object appears
        first_time = object_times[0]
        
        # Find the closest image index for this time
        closest_idx = np.abs(self.globalTimes - first_time).argmin()
        
        # Snap to the first frame where the object appears
        self.time_change_lock = True
        self.current_time_value = first_time
        self.current_image_index = closest_idx
        self.ac_view.setCurrentIndex(closest_idx)
        self.dc_view.setCurrentIndex(closest_idx)
        self.time_change_lock = False
        
        # Update contours for the selected object
        self.update_contours()
    
    def clear_contours(self):
        """Clear all contour items from both views."""
        for view_name in ['ac', 'dc']:
            for item in self.contour_items[view_name]:
                if view_name == 'ac':
                    self.ac_view.getView().removeItem(item)
                else:
                    self.dc_view.getView().removeItem(item)
            self.contour_items[view_name] = []
    
    def update_contours(self):
        """Update contours for the current object and time."""
        if not self.current_object:
            return
            
        obj_data = self.data_dict.get(self.current_object, {})
        if not obj_data:
            return
            
        # Get the list of time values where this object appears
        object_times = obj_data.get('times', [])
        if not len(object_times):
            return
        
        # Convert to a list if it's a numpy array
        if isinstance(object_times, np.ndarray):
            object_times = object_times.tolist()
            
        # Find the closest time in the object's timeline to the current time
        closest_time_idx = np.abs(np.array(object_times) - self.current_time_value).argmin()
        object_time = object_times[closest_time_idx]
        
        # Only draw contours if we're within a reasonable tolerance of the object's time
        # This avoids drawing contours when the object doesn't actually exist at the current time
        time_tolerance = 0.1  # Adjust this value as needed (in minutes)
        if abs(object_time - self.current_time_value) <= time_tolerance:
            # Draw interior contours (green) - can be multiple contours per timepoint
            interior_contours = obj_data.get('Interior contour', [])
            if interior_contours and closest_time_idx < len(interior_contours):
                current_interior_contours = interior_contours[closest_time_idx]
                
                # Handle both cases: single contour or list of contours
                if isinstance(current_interior_contours, list) or isinstance(current_interior_contours, tuple):
                    # Handle multiple contours per timepoint
                    for contour in current_interior_contours:
                        self.draw_contour(contour, 'green', 'ac')
                        self.draw_contour(contour, 'green', 'dc')
                else:
                    # Handle single contour as numpy array
                    self.draw_contour(current_interior_contours, 'green', 'ac')
                    self.draw_contour(current_interior_contours, 'green', 'dc')
            
            # Draw total contour (red) - single contour per timepoint
            total_contours = obj_data.get('Total contour', [])
            if total_contours and closest_time_idx < len(total_contours):
                self.draw_contour(total_contours[closest_time_idx], 'red', 'ac')
                self.draw_contour(total_contours[closest_time_idx], 'red', 'dc')
    
    def draw_contour(self, contour_points, color, view_type):
        """
        Draw a contour on the specified view.
        
        Parameters:
        -----------
        contour_points : numpy.ndarray
            Nx2 array of contour points
        color : str
            Color of the contour
        view_type : str
            'ac' or 'dc' to specify which view to draw on
        """
        if contour_points is None or len(contour_points) == 0:
            return
            
        # Create a plot curve item for the contour
        contour_item = pg.PlotCurveItem(
            x=contour_points[:, 0],
            y=contour_points[:, 1],
            pen=pg.mkPen(color=color, width=2)
        )
        
        # Add the contour to the appropriate view
        if view_type == 'ac':
            self.ac_view.getView().addItem(contour_item)
            self.contour_items['ac'].append(contour_item)
        else:
            self.dc_view.getView().addItem(contour_item)
            self.contour_items['dc'].append(contour_item)


# Example usage:
if __name__ == "__main__":
    import sys
    
    # Example data with multiple interior contours per timepoint
    example_data = {
        "Object1": {
            "idx": [0, 1, 2],  # This object appears in frames 0, 1, and 2
            "Interior contour": [
                # Frame 0: multiple contours
                [
                    np.array([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]),
                    np.array([[30, 30], [40, 30], [40, 40], [30, 40], [30, 30]])
                ],
                # Frame 1: multiple contours
                [
                    np.array([[15, 15], [25, 15], [25, 25], [15, 25], [15, 15]]),
                    np.array([[35, 35], [45, 35], [45, 45], [35, 45], [35, 45]])
                ],
                # Frame 2: multiple contours
                [
                    np.array([[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]),
                    np.array([[40, 40], [50, 40], [50, 50], [40, 50], [40, 40]])
                ]
            ],
            "Total contour": [
                np.array([[5, 5], [45, 5], [45, 45], [5, 45], [5, 5]]),        # for frame 0
                np.array([[10, 10], [50, 10], [50, 50], [10, 50], [10, 10]]),  # for frame 1
                np.array([[15, 15], [55, 15], [55, 55], [15, 55], [15, 15]])   # for frame 2
            ]
        },
        "Object2": {
            "idx": [3, 4, 5],  # This object appears in frames 3, 4, and 5
            "Interior contour": [
                # Frame 3: multiple contours
                [
                    np.array([[50, 50], [60, 50], [60, 60], [50, 60], [50, 50]]),
                    np.array([[70, 70], [80, 70], [80, 80], [70, 80], [70, 70]])
                ],
                # Frame 4: single contour
                np.array([[55, 55], [75, 55], [75, 75], [55, 75], [55, 55]]),
                # Frame 5: multiple contours
                [
                    np.array([[60, 60], [70, 60], [70, 70], [60, 70], [60, 60]]),
                    np.array([[80, 80], [90, 80], [90, 90], [80, 90], [80, 80]])
                ]
            ],
            "Total contour": [
                np.array([[40, 40], [90, 40], [90, 90], [40, 90], [40, 40]]),  # for frame 3
                np.array([[45, 45], [85, 45], [85, 85], [45, 85], [45, 45]]),  # for frame 4
                np.array([[50, 50], [95, 50], [95, 95], [50, 95], [50, 95]])   # for frame 5
            ]
        }
    }
    
    # Create example image data (6 frames)
    ac_images = [np.random.rand(100, 100) for _ in range(6)]
    dc_images = [np.random.rand(100, 100) for _ in range(6)]
    
    app = QtWidgets.QApplication(sys.argv)
    window = CellViewer(example_data, ac_images, dc_images)
    window.show()
    sys.exit(app.exec_())