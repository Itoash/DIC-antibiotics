# Description and Features

## Brief principle
This applet is meant to analyze the per-pixel signal in (phase-contrast/DIC) images, obtained by applying an oscillating electric field during hihg-speed acquisition. This is usually done to study the electrical properties of the cell, as the electric filed modulates the refractive index change across interfaces, meaning this is especially useful to study the structure of cell membranes, organelles, and such.

## Function of the program
This app is meant to provide a user interface for the processing and obtaining single-cell measurements from electro-optical time series. Obtaining the data can be broken down into four important steps:
* Signal analysis
* Cell segmentation
* Cell tracking
* Single cell measurements

### Signal analysis
In an electro-optical measurement stack, each spatial location has its associated time series, which mirrors the AC signal applied during acquisition (provided the field is strong enough, and there are refractive interfaces to be found in the focus plane). Using traditional signal processing methods to clean up the signal and extract the **amplitude** of the desired frequency from the signal, the app then provides an "Amplitude image" to the user, indicating where the oscillations were strongest. 
An experiment will ususally study the progression of the electrical properties of these interfaces over time, usualy under the effect of some external stimuli. This means that a large number of images (measurements of +/- 400 frames, over however many measurements are ob### Signal Analysis

In an electro-optical measurement stack, each spatial location has its associated time series, mirroring the AC signal applied during acquisition. The app uses signal processing methods to clean up the signal and extract the **amplitude** of the desired frequency, providing an "Amplitude image" to indicate where oscillations were strongest.

#### Key Features:
- **Parallel Signal Processing:** Efficiently processes large image stacks (~400 frames per measurement).
- **Amplitude Image:** Highlights regions with the strongest oscillations.

#### Workflow:
1. Apply signal processing methods.
2. Extract amplitude of the desired frequency.
3. Generate amplitude images for analysis.tained) need to be processed in a reasonable time. This is where the highly-efficient parallel signal processing comes in (details later). 

### Segmentation
Using a pre-existing model (Omnipose, cite here), tuned to DIC images when necessary, the program performs segmentations on the (time-averaged) image stack, obtaining cell masks with great accuracy, especially considering the quality of the images used; DIC microscopy doesn't show a great difference in intensity on objects in focus. The segmentation model is pre-trained with a large number of phase-contrast images, but adapting some of its parameters allows it to make decently accurate predictions on DIC images as well, provided the cells have some contrast from the background.

### Tracking
From the segmentation image series, (remember, multiple measurements over time), a tracking networ is constructed by evaluating similarities between cells, and considering biological aspects like mitosis. Currently, merges between masks are disallowed, as they are usually either a result of overlapping cells or loss of contrast in the source images. The tracking panel provides the user with a large number of tools for editing/correcting segmentation and tracking issues. These include but are not limited to:
* Editable segmentation masks: MSPaint-style with drawing or by changing colors/deleting entire cells
* Filtering based on size/convexity/aspect ratio
* Editing lineages manually, or removing entire branches from the lineage tree at once

Though not very sophisticated in GUI terms, the interface provided at this stage is functional and prevents the user from **severely** shooting themselves in the foot in most cases (but not all, as we'll see in the user guide).

### Single cell measurements
Before this project was finalized, which observables were relevant to the cells' evolution were determined empirically, and those are now hard-coded into the app. Measurmenets are done algorithmically for each cell, in parallel for speed, and displayed to the user as a time series, where they can be easily compared to the others to study interesting dynamics.

The user has some measure of control only on the segmentation and tracking results, because those steps are indeed fallible, and there exists an objectively correct output (correct segmentation masks and lineage reconstructions/respectively) that the program will most likely **not** perfectly reach. For completely error-free data, there has to be a user in the loop.



# Installation
Most of the installation steps are the same on all platforms:
* Install a version of a Python interpreter using your method of choice (see Python's [beginner's guide to installing Python](https://wiki.python.org/moin/BeginnersGuide/Download) for info)
* Install a C/C++ compiler, for using Cython:
    * On <span style="color:beige">Windows</span> this is usually included in the Visual Studio Community Edition (see downloads [here](https://visualstudio.microsoft.com/vs/community/)), then select the C++ Development Kit during installation and it should be good to go
    * On <span style="color:beige">Mac</span> and <span style="color:beige">Linux</span> this usually comes prepackaged

* Install Git on your platform (see official documentation and install guides here [here](https://git-scm.com))
* **On CUDA-enabled machines:** Install a recent (12.x) version of Nvidia's CUDA Toolkit for GPU processing and segmentation
* On your machine, create and activate a [Python virtual environment](https://docs.python.org/3/library/venv.html) using your method of choice
* Download the archive and move it to the virtual environment folder

* Use the command:
 ```
 pip install <archive-name>
```
* Wait for the process to finish, then type '**startGUI**' to start the application.


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
* **Update analysis**: redo signal processing with selected parameters
* **Reset analysis**: redo signal processing with default parameters
* **Frequency**: which frequency would you like ot extract?
* **Interpolate**: interpolate signal so Fourier spectrum is guaranteed to contain EXACTLY the frequency you want
* **Fitler**: Apply bandpass to filter out some of the unnecessary frequencies
* **Filter limits**: sets the filter limits (obviously). won't let you set below 0 or higher than the Nyquist frequency.

there's also an interactive region bar on the **Signal** plot you might have noticed, on account of the fact that it's big and blue. Moving the ends of that region allows you to specify which part of the signal you want to analyze, but on each subsequent update the amount of data shrinks and you have to reset the analysis to get back to where you were. We'll call this a feature not a problem.

The signal pane will update in real time as you move the ROIs along in any of the other panes, displaying the mean signal and FFT for the region enclosed by the ROI. It will reset when you click out of the ROI menu.









