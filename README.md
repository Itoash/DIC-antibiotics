

## Table of Contents

1. [Description and Features](#description-and-features)
    - [Brief Principle](#brief-principle)
    - [Function of the Program](#function-of-the-program)
        - [Signal Analysis](#signal-analysis)
        - [Segmentation](#segmentation)
        - [Tracking](#tracking)
        - [Single Cell Measurements](#single-cell-measurements)
2. [Installation](#installation)
3. [User Guide](#user-guide)
    - [Main Window](#1-main-window)
        - [File Tree](#1a-the-file-tree)
        - [AC Pane](#1b-the-ac-pane)
        - [DC Pane](#1c-the-dc-pane)
        - [Raws Pane](#1d-the-raws-pane)
        - [Signals Section](#1e-the-signals-section)
4. [Troubleshooting](#Troubleshooting)

    - [Installation errors](#installation-errors)
    - [Error messages](#error-messages)
    - [Undocumented errors and the debug console](#undocumented-errrors-and-the-debug-console)
5. [Appendix](#appendix)
    - [Python virtual environments](#python-virtual-environments)
    - [Naming conventions](#file-naming-conventions-and-other-headaches)

# Description and Features

## Brief principle

This applet is meant to analyze the per-pixel signal in (phase-contrast/DIC) images, obtained by applying an oscillating electric field during hihg-speed acquisition. This is usually done to study the electrical properties of the cell, as the electric filed modulates the refractive index change across interfaces, meaning this is especially useful to study the structure of cell membranes, organelles, and such.

## Function of the program

This app is meant to provide a user interface for the processing and obtaining single-cell measurements from electro-optical time series. Obtaining the data can be broken down into four important steps:

1. Signal analysis
2. Cell segmentation
3. Cell tracking
4. Single cell measurements

### Signal Analysis

In an electro-optical measurement stack, each spatial location has its associated time series, mirroring the AC signal applied during acquisition. The app uses signal processing methods to clean up the signal and extract the **amplitude** of the desired frequency, providing an "Amplitude image" to indicate where oscillations were strongest.

#### Signal Analysis Key Features

- **Parallel Signal Processing:** Efficiently processes large image stacks (~400 frames per measurement).
- **Amplitude Image:** Highlights regions with the strongest oscillations.
- **Signal Examination:** Signals from specific regions can be examined individually after analysis.

#### Workflow:

1. Apply signal processing methods.
2. Extract amplitude of the desired frequency.
3. Generate amplitude images for analysis.
4. Examine and modify signal processing results.

### Segmentation

Using a pre-existing model (Omnipose, cite here), tuned to DIC images when necessary, the program performs segmentations on the (time-averaged) image stack, obtaining cell masks with great accuracy, especially considering the quality of the images used; DIC microscopy doesn't show a great difference in intensity on objects in focus. The segmentation model is pre-trained with a large number of phase-contrast images, but adapting some of its parameters allows it to make decently accurate predictions on DIC images as well, provided the cells have some contrast from the background.

#### Key Features:

- **Fast, GPU-based segmentation:** Using PyTorch's backend selection, images can be processed quickly on most devices.
- **Editable segmentation masks:** In the dedicated segmentation panel, masks can be edited and modified using drawing tools or bulk operations (delete label x, change label y to brush color etc.).
- **(Coming soon) Hot-swapping between phase contrast/DIC:** To enable greater segmentation flexibility, a segmentation mode selector will be added for switching between DIC and pahse contrast images.

#### Workflow

1. Segment images of interest
2. Open Segmentation Editor
3. Check and correct masks

### Tracking

From the segmentation image series, (remember, multiple measurements over time), a tracking networ is constructed by evaluating similarities between cells, and considering biological aspects like mitosis. Currently, merges between masks are disallowed, as they are usually either a result of overlapping cells or loss of contrast in the source images. The tracking panel provides the user with a large number of tools for editing/correcting segmentation and tracking issues. These include but are not limited to:

#### Key Features:

- **Robust, automatic tracking:** Hybrid IoU/shape features tracker, with additional considerations for biology (cell mitosis). Provided the masks are assigned correctly, the tracker provides accurate lineage reconstruction with minimal human interference.
- **Editable segmentation masks:** Mirroring the segmentation panel, masks can be edited and modified using drawing tools or bulk operations (delete label x, change label y to brush color etc.). Filtering based on cell features is also available.
- **Editable cell lineages:** In the (hopefully) rare cases where the tracker does fail, lineage correction is possible via interacting with the lineage graph panel: linking/unlinking nodes, as well as removing entire lineage branches from the graph and images, is possible.
- **Image stabilization (experimental):** Image stabilization can be applied to the image data obtaine, using the masks as a guide. THIS WILL CHANGE THE IMAGE DATA PERMANENTLY!

#### Workflow:

1. Track segmentation masks automatically
2. Correct erroneous masks/links
3. Continue analyzing the data

####

Though not very sophisticated in GUI terms, the interface provided at this stage is functional and prevents the user from **severely** shooting themselves in the foot in most cases.

### Single cell measurements

Before this project was finalized, which observables were relevant to the cells' evolution were determined empirically, and those are now hard-coded into the app. Measurmenets are done algorithmically for each cell, in parallel for speed, and displayed to the user as a time series, where they can be easily compared to the others to study interesting dynamics.

#### Key Features:

- **Fast Per-Cell Feature Computation:** Using parallel processing, cells are analyzed in all image types by using the provided segmentation masks and obtained lineage graph.
- **Easy Cell Contour Visualisation:** In both AC (amplitude) and DC(time-averaged) images, the user can view the obtained cell contours and spot issues.
- **Easy Cell Feature Plotting:** The Time Series Viewer panel provides a convenient way to quickly chekc the progression of each cell through time, on any of the computed features. It also allows comparison with all of the viewed cells, as well as cells in a selected subset or particular lineage.

- **Data Saving:** Most importantly, at any point during the analysis process, obtained cell data can be saved, either for external analysis or to continue the analysis later.

#### Workflow:

1. Analyse data per cell
2. Visualise results
3. Save to disk


The user has some measure of control only on the segmentation and tracking results, because those steps are indeed fallible, and there exists an objectively correct output (correct segmentation masks and lineage reconstructions/respectively) that the program will most likely **not** perfectly reach. For completely error-free data, there has to be a user in the loop.

# Installation

Follow these steps to install the application:

1. Install Python:
   - Refer to Python's [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download).

2. Install a C/C++ Compiler:
   - **Windows:** Use Visual Studio Community Edition ([Download](https://visualstudio.microsoft.com/vs/community/)).
   - **Mac/Linux:** Usually pre-installed.

3. Install Git:
   - Follow the [official documentation](https://git-scm.com).

4. **CUDA-enabled machines:** Install Nvidia's CUDA Toolkit (version 12.x).

5. Create and activate a [Python virtual environment](https://docs.python.org/3/library/venv.html).

6. Download the archive and move it to the virtual environment folder.

7. Install the application:

   ```bash
   pip install <archive-name>
   ```

8. Wait for the process to finish, then type '**startGUI**' to start the application.

# Quickstart

A normal workflow usually consists of:

1. Acquire set of measurements, from one or multiple locations in a sample
2. Save measurements in a common folder
3. Open measurement time series, **one at a time**.
4. Process signals, segment and track, and view results
5. Save cell data

For this workflow a "quickstart guide" for the anlaysis part would run as follows:

1. Open the application:
    -Via either the desktop shortcut if installed (labelled *AnalysisGUI*), or by entering the app's [virtual environment](#python-virtual-environments)  and running the command:

    ```bash
    startGUI
    ```

2. You'll be greeted by the main window screen; you can now try out the signal processing by selecting a measurement folder (either an Andor Solis spool folder or a TIFF images folder following the [naming convention](#file-naming-conventions-and-other-headaches)) and entering the **Load Stack/Load Spool** command from the file menu, or **Ctrl+L/Ctrl+P** as a shortcut.

    <div style="text-align: center;">
    <img src="./assets/start_screen.png" alt="Main Splash Screen" />
    <p><em>The main splash screen displayed upon application startup.</em></p>
    </div>


3. The results will be displayed in the AC, DC and Raws plots of the main window. The mean signal and Fourier Transform will be shown in the Signals plot. From the Signals plot multiple signal processing options can be selected for further tailoring the analysis; for now, the only important parameter is the frequency.
    <div style="text-align: center;">
    <img src="./assets/signal_pane.png" alt="Signals Panel" />
    <p><em>The Signals panel. All the options for signal processing are located here. The <b>Frequency</b> field has to match the applied frequency during the experiment. </em></p>
    </div>

    After ensuring you have the correct frequency locked in, hit **Update Analysis** to view the new results. The panels will update to show the processed results.

    <div style="text-align: center;">
    <img src="./assets/loaded_stack.png" alt="Loaded Image Data" />
    <p><em>Loaded microscopy data. The <b>amplitude (AC)</b> image is visible in the centre, while the <b>time-averaged (DC)</b> and <b>Raw</b> image data is nested in the panel on the top right. The <b>Signals</b> plot now displays the average signal and FFT for the loaded measurement. </em></p>
    </div>

4. When satisfied with the signal analysis results, you'll want to load the entire experiment for analysis. You can do that by selecting the relevant experiment folder in the file explorer on the left, then using either the **Load Folder of Spools** or **Load Folder of Stacks** option in the **File** menu (or by hitting either Ctrl+? or Ctrl+A, respectively). This will start the batch loading of images, automatically applying the same signal processing steps you set after you hit **Update Analysis**, segmenting the resulting DC images, and storing them in order, in a buffer. Refer to the current nmaming convention in the Appendix (link appendix) for information on how the program expects TIFF folders to be called.

5. After loading is complete, you are free to select **Start Segmentation** or **Start Tracking** from the **Segmentation** and **Tracking** menus, respectively. In general, the tracking menu performs a lot of the same functionality as the segmentation menu, with the added bonus of manipulating cell lineages and filtering masks based on cell features; thus it's recommended to skip directly to tracking if the masks are good enough. 
    <div style="text-align: center;">
    <img src="./assets/tracking_menu.png" alt="Start Tracking button" />
    </div>

6. The Tracking window displays an overlaid mask/DC image (top) and a lineage graph (bottom).
    <div style="text-align: center;">
    <img src="./assets/tracking_window.png" alt="Tracking Panel" />
    <p><em>The Tracking window. The currently selected node in the graph is highlighted in the overlay. The graph can be navigated via the arrow keys, and opacity of the overlay can be adjusted by using the slider in the toolbar. </em></p>
    </div>

    The masks can be edited using the tools in the top left, as well as right-clicking to delete/change labels of entire masks at once. The lineage graph can be manipulated via right-clicking, and can link/unlink nodes or delete branches.

    <div style="text-align: center;">
    <img src="./assets/drawing_tools.png" alt="Tracking Panel" />
    <p><em>Drawing tools for editing masks. </em></p>
    </div>

    <div style="text-align: center;">
    <img src="./assets/linking_tools.png" alt="Tracking Panel" />
    <p><em>Lineage editing tools for editing lineages. Also displays measurements for the currently selected cell. </em></p>
    </div>

    The entire tracking stage works via a checkpoint system: you have **Checkpoint** and **Reset** buttons for saving/loading the images and the tracking network. Checkpointing overwrites the previously saved data, so be careful with edits. There is no undo button (yet, mainly due to the drawing steps)!The **Recompute Network** button also has to be periodically pressed when editing masks to feed the new segmentation data into the tracker and keep the lineage graph up to date.  When you are finished working on the network, hit **Checkpoint** one last time and you can now visualise the results.

    <div style="text-align: center;">
    <img src="./assets/visualisation_buttons.png" alt="Buttons for opening visualisations" />
    <p><em>Buttons for opening visualisations. </em></p>
    </div>

7. Hitting either **Open Cell Visualiser** or **Open Time Series Viewer** will take you to two data visualsiation windows: one for visualising the cells' detected contours in both AC and DC images:
    <div style="text-align: center;">
    <img src="./assets/cell_vis_window.png" alt="CellVis" />
    <p><em>Cell visualisation window. Draws interior and external contours on both AC and DC images. Multiple cells can be seelcted for simlutanous viewing. </em></p>
    </div>
    The other for viewing the evolution of cell parameters over time, individually or as a group:
    <div style="text-align: center;">
    <img src="./assets/time_series_window.png" alt="CimeSeries" />
    <p><em>Time Series display with all cells plotted. Time range of interest can be seelcted using the sliding window on the bottom plot. Multiple selection modes (single-cell, multi-cell, lineage, all cells) are available. </em></p>
    </div>

 

8. The final step is to save the work. By hitting **Ctrl+S** (or **Save** int the **File** menu) you can choose a location to save your cell data.

The current format for cell data is a bit atypical, mainly for space-saving reasons, but it can be imported and used fairly easily in any standard data analysis tool (Excel, Python w/ Pandas, R etc.). And that's it.


# User Guide

## 1.) Main Window

This is the first window shown at startup. There are several different sections to choose from, so we'll go over them separately, then return to the main menu to explore how this window links to the other components of the program.

###  1.a) The file tree

Because the program is bound to do a lot of I/O and inspecting different measurement stacks, a file tree is included on the left-most side of the window. This doesn't do much in and of itself, but serves as an "index" of sorrts for the rest of the program: when calling most of the I/O functions, the currently selected index is where they will turn to first. For instance **(insert example with image)**.
This makes switching between measurements or entire experiments easy and fast, as it doesn;t require the user to enter a file selection dialog, and it makes the entire set of available data visible.

###  1.b) The AC Pane

The AC pane is where the user can check the results of the signal processing step. It can display values per pixel (via hovering with the moouse), or averaged per region (using the ROI button in the bottom left, and dragging the box around). The averaged results (on the horizontal axis, a sort of intensity profile) are displayed at the bottom of the image. Be aware that messing with the ROI button will cause the displayed signals to change as well.

###  1.c) The DC Pane

Extremely similar (some would say identical) to the AC pnae, only displaying the time-averaged image stack. Interact with it as you would the AC pane

###  1.d) The Raws Pane

Hiding behind the DC pane is the raw images pane. This is a bit more interesting than the previous ones, as it actually allows you to view each image in the stack individually.

### 1.e) The Signals Section

Main output and controller of the signal processing step (really, it should be first thing you see, not tucked away in the bottom-right corner). It provides methods for tweaking the signal processing, and a (sometimes live) view of the mean signal and FFT obtained from the image stack. Main options include:

- **Update analysis**: redo signal processing with selected parameters
- **Reset analysis**: redo signal processing with default parameters
- **Frequency**: which frequency would you like ot extract?
- **Interpolate**: interpolate signal so Fourier spectrum is guaranteed to contain EXACTLY the frequency you want
- **Fitler**: Apply bandpass to filter out some of the unnecessary frequencies
- **Filter limits**: sets the filter limits (obviously). won't let you set below 0 or higher than the Nyquist frequency.

there's also an interactive region bar on the **Signal** plot you might have noticed, on account of the fact that it's big and blue. Moving the ends of that region allows you to specify which part of the signal you want to analyze, but on each subsequent update the amount of data shrinks and you have to reset the analysis to get back to where you were. We'll call this a feature not a problem.

The signal pane will update in real time as you move the ROIs along in any of the other panes, displaying the mean signal and FFT for the region enclosed by the ROI. It will reset when you click out of the ROI menu.

# Troubleshooting

## Installation errors

## Error messages

## Undocumented errors and the debug console

# Appendix

## Python virtual environments
[Google it!](https://realpython.com/python-virtual-environments-a-primer/)

## File naming conventions and other headaches