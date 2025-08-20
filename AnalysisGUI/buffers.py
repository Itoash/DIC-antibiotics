from pyqtgraph.Qt import QtWidgets
import numpy as np 
import os
from cellpose import models
import requests
import zipfile
class AnalysisBuffer:
    """
    Class for long-term storage of DC and AC images, as well as metadata.

    Stores AC/DC images, unique identifier names, and timestamps.
    Used for cell-by-cell analysis and displaying results.
    Optionally links to a SegmentationBuffer for synchronized sorting.
    """

    def __init__(self, segBuffer=None):
        """
        Initializes the AnalysisBuffer.

        Parameters
        ----------
        segBuffer : SegmentationBuffer, optional
            Reference to a SegmentationBuffer for synchronized sorting.
        """
        self.ACs = []
        self.DCs = []
        self.names = []
        self.times = []
        self.abstimes = []
        self.segBuffer = segBuffer

    def addImage(self, AC, DC, name, resort=True):
        """
        Adds AC and DC images with associated metadata.

        Parameters
        ----------
        AC : np.ndarray
            AC image loaded from AC Analysis tab.
        DC : np.ndarray
            DC image loaded from AC Analysis tab.
        name : str
            Directory name for origin of raw images.
        resort : bool, optional
            Whether to sort the buffer by absolute time after adding (default is True).

        Returns
        -------
        None
        """
        self.ACs.append(AC.astype(np.float64))
        self.DCs.append(DC.astype(np.float64))
        self.names.append(name)
        time, abstime = self.figureTime(name)
        self.times.append(time)
        self.abstimes.append(abstime)
        if resort:
            self.sortByAbsTime()

    def sortByAbsTime(self):
        """
        Sorts all stored data (and linked segmentation buffer, if present) by absolute time.

        Returns
        -------
        None
        """
        if self.segBuffer is not None:
            zipped = list(zip(self.abstimes, self.times, self.names, self.ACs, self.DCs, self.segBuffer.images, self.segBuffer.masks))
            zipped.sort(key=lambda x: x[0])
            self.abstimes, self.times, self.names, self.ACs, self.DCs, self.segBuffer.images, self.segBuffer.masks = map(list, zip(*zipped))

    def figureTime(self, name):
        """
        Extracts time information from the filename.

        Assumes naming convention:
        <magnification>_<method>_<voltage>_<ms>_<fps>_-<s/supr>_<date YYY/MM/DD>_<time 12H fmt> <AM/PM>

        Parameters
        ----------
        name : str
            Full filename of loaded directory.

        Returns
        -------
        time : tuple
            (hour, minutes) in 24h format.
        abstime : float
            Absolute time in minutes.
        """
        timestring = name.split('_')[-1]
        ext = timestring.split(' ')[-1]
        timestring = timestring.split(' ')[0]
        if ext.upper() == 'AM':
            hours = int(timestring[0:2])
            minutes = int(timestring[2:4])
            seconds = int(timestring[4:6])
        elif ext.upper() == 'PM':
            if len(timestring) == 5:
                hours = int(timestring[0]) + 12
                minutes = int(timestring[1:3])
                seconds = int(timestring[3:5])
            else:
                hours = int(timestring[0:2])
                minutes = int(timestring[2:4])
                seconds = int(timestring[4:6])

        time = (hours, minutes)
        abstime = 60 * hours + minutes + seconds / 60
        return time, abstime

    def clear(self):
        """
        Wipes all internal info clean.

        Returns
        -------
        None
        """
        self.ACs = []
        self.DCs = []
        self.names = []
        self.times = []
        self.abstimes = []

MODEL_URLS = {
    "cell_model": "https://github.com/Itoash/DIC-antibiotics/releases/download/v1.0/cpsam-DIC.zip",
    "bact_model": "https://github.com/Itoash/DIC-antibiotics/releases/download/v1.0/cpsam-DIC-bact.zip",
}

PERSISTENT_MODEL_DIR = os.path.expanduser("~/.analysisgui/models")

CELL_MODEL_PATH = os.path.join(PERSISTENT_MODEL_DIR, 'cpsam-DIC')
BACT_MODEL_PATH = os.path.join(PERSISTENT_MODEL_DIR, 'cpsam-DIC-bact')

def download_and_unzip_model(url, output_dir):
    """
    Downloads and unzips a model from the given URL.

    Parameters
    ----------
    url : str
        URL of the zipped model file.
    output_dir : str
        Directory to store the unzipped model.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, os.path.basename(url))

    # Download the file if it doesn't already exist
    if not os.path.exists(zip_path):
        print(f"Downloading model from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            # Get the total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Download progress: {percent:.1f}%", end='\r')
            
            print(f"\nDownloaded model to {zip_path} ({downloaded} bytes)")
            
            # Verify the downloaded file is a valid zip
            if not zipfile.is_zipfile(zip_path):
                print(f"Error: Downloaded file is not a valid zip file. Removing {zip_path}")
                os.remove(zip_path)
                return
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return

    # Check if model is already extracted
    model_name = os.path.basename(url).replace('.zip', '')
    extracted_path = os.path.join(output_dir, model_name)
    if os.path.exists(extracted_path):
        print(f"Model already extracted at {extracted_path}")
        return

    # Unzip the file
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
            print(f"Unzipped model to {output_dir}")
            
        # Clean up the zip file after successful extraction
        os.remove(zip_path)
        print(f"Removed zip file {zip_path}")
        
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file or is corrupted")
        os.remove(zip_path)
    except Exception as e:
        print(f"Error extracting zip file: {e}")


class SegmentationBuffer:
    """
    Buffer for storing DC and segmentation image data.

    Stores images and their corresponding segmentation masks.
    """

    def __init__(self,eukaryotes=False):
        """
        Initializes the SegmentationBuffer.
        """
        self.images = []
        self.masks = []
        self.imagenet = None
        self.checkModels()
        if eukaryotes:
            self.model = models.CellposeModel(gpu=True, pretrained_model=CELL_MODEL_PATH,use_bfloat16=False)
            self.eukaryotic_mode = True
        else:
            self.model = models.CellposeModel(gpu=True, pretrained_model=BACT_MODEL_PATH,use_bfloat16=False)
            self.eukaryotic_mode = False

    def checkModels(self):
        """
        Checks if models exist and downloads them if they don't.
        """
        
        cell_model_exists = os.path.exists(CELL_MODEL_PATH)
        bact_model_exists = os.path.exists(BACT_MODEL_PATH)
        
        if not cell_model_exists:
            print(f"Cell model not found at {CELL_MODEL_PATH}")
            print(f"Downloading cell model from {MODEL_URLS['cell_model']}...")
            download_and_unzip_model(MODEL_URLS['cell_model'], PERSISTENT_MODEL_DIR)
        
        if not bact_model_exists:
            print(f"Bacterial model not found at {BACT_MODEL_PATH}")
            print(f"Downloading bacterial model from {MODEL_URLS['bact_model']}...")
            download_and_unzip_model(MODEL_URLS['bact_model'], PERSISTENT_MODEL_DIR)
        
        # Verify models exist after download
        if not os.path.exists(CELL_MODEL_PATH):
            print(f"Warning: Cell model still not found at {CELL_MODEL_PATH}")
        if not os.path.exists(BACT_MODEL_PATH):
            print(f"Warning: Bacterial model still not found at {BACT_MODEL_PATH}")

    def setModel(self, eukaryotes=False):
        """
        Sets the model for segmentation.

        Parameters
        ----------
        eukaryotes : bool, optional
            If True, uses the eukaryotic model; otherwise, uses the bacterial model.
        
        Returns
        -------
        None
        """
        if eukaryotes == self.eukaryotic_mode:
            # No change needed, model already set correctly
            return
        if eukaryotes:
            self.model = models.CellposeModel(gpu=True, pretrained_model=CELL_MODEL_PATH,use_bfloat16=False)
            self.eukaryotic_mode = True
        else:
            self.model = models.CellposeModel(gpu=True, pretrained_model=BACT_MODEL_PATH,use_bfloat16=False)
            self.eukaryotic_mode = False
    
    def addImage(self, img,params = None):
        """
        Adds a single image and computes its segmentation mask.

        Parameters
        ----------
        img : np.ndarray
            Image to add.

        Returns
        -------
        None
        """
        if len(self.images) != 0:
            shapelist = [im.shape for im in self.images]
            if img.shape != shapelist[0]:
                QtWidgets.QApplication.restoreOverrideCursor()
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText("Cannot add image!")
                msg.setInformativeText('Wrong dimensions')
                msg.setWindowTitle("TypeError")
                return

        self.images.append(img.astype(np.float64))
        if params is not None:
            # If parameters are provided, use them for segmentation
            # Remove eukaryotic_mode from params as it's not a valid argument for model.eval
            model_params = {k: v for k, v in params.items() if k != 'eukaryotic_mode'}
            msk,*_ = self.model.eval([img],**model_params)

        else:
            msk,*_ = self.model.eval([img], diameter=None, flow_threshold=0.0, cellprob_threshold=0.0, do_3D=False, niter=None, augment=False)

        self.masks.append(msk[0])

    # def addMultipleImages(self, imgs):
    #     """
    #     Adds multiple images and computes their segmentation masks.

    #     Parameters
    #     ----------
    #     imgs : list of np.ndarray
    #         List of images to add.

    #     Returns
    #     -------
    #     None
    #     """
    #     shapelist = set([im.shape for im in imgs])
    #     if len(shapelist) != 1:
    #         QtWidgets.QApplication.restoreOverrideCursor()
    #         msg = QtWidgets.QMessageBox()
    #         msg.setIcon(QtWidgets.QMessageBox.Critical)
    #         msg.setText("Different sizes!")
    #         msg.setInformativeText(
    #             'Images have varying sizes. Select image folder with same size images throughout.')
    #         msg.setWindowTitle("TypeError")
    #         return
    #     if len(self.images) != 0:
    #         if shapelist[0] != self.images[0].shape:
    #             QtWidgets.QApplication.restoreOverrideCursor()
    #             msg = QtWidgets.QMessageBox()
    #             msg.setIcon(QtWidgets.QMessageBox.Critical)
    #             msg.setText("Wrong sizes!")
    #             msg.setInformativeText(
    #                 'Images have different sizes than those already in buffer!')
    #             msg.setWindowTitle("TypeError")
    #             return

    #     msk = segmentDComni(imgs)
    #     self.masks += msk
    #     self.images += imgs

    def clear(self):
        """
        Clears all stored images and masks.

        Returns
        -------
        None
        """
        self.images = []
        self.masks = []

    def remove(self, index):
        """
        Removes an image and its mask at the specified index.

        Parameters
        ----------
        index : int
            Index of the image/mask to remove.

        Returns
        -------
        None
        """
        self.images.pop(index)
        self.masks.pop(index)

def test_url_access():
    """Test if the URLs are accessible"""
    for name, url in MODEL_URLS.items():
        try:
            response = requests.head(url, timeout=10)
            print(f"{name}: Status {response.status_code}")
            if response.status_code == 200:
                print(f"  Content-Length: {response.headers.get('content-length', 'Unknown')}")
                print(f"  Content-Type: {response.headers.get('content-type', 'Unknown')}")
        except Exception as e:
            print(f"{name}: Error - {e}")