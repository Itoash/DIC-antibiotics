#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.stats as st
from copy import deepcopy
from numba import njit
from lmfit import Model
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.signal import lombscargle
from scipy.fft import fft, fftfreq, ifft, fftshift
from scipy import interpolate
from scipy.signal.windows import hamming

#%%



# ROIAC_15oct = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 15 oct antibiotic/ResultsAC15oct.csv')
# times_15oct = [0,8,18,36,49,75,85,109,130,153,181,241]
# oct15_dims = (278,288)

# ROIAC_17feb = pd.read_csv("/Users/victorionescu/Desktop/imagini si rezultate/17 februarie 25 CST/ResultsCorr.csv")
# times_17feb = [0, 2, 3, 7, 8, 11, 15, 16, 17, 20, 23, 27, 31,  35, 39, 42, 51, 53, 61, 65, 66, 69, 72, 74, 76, 79, 80, 81, 85, 87, 89, 90, 93, 96, 97, 99, 102, 105, 107, 108, 109, 110, 113, 118, 126, 128, 131, 135]
# feb17_dims = (372,376)
# feb17_bckg = [1 for x in times_17feb]

# ROIAC_12jul = pd.read_csv("/Users/victorionescu/Desktop/imagini si rezultate/12 jul 24 CST/ResultsCorr.csv")
# times_12jul = [0, 6, 12, 17, 21, 26, 34, 40, 45, 61, 67, 75, 91, 97, 105, 113, 120, 136, 138, 151, 161]
# print(len(times_12jul))
# jul12_dims = (323,496)
# jul12_bckg = [1 for x in times_12jul]







# ROIAC_4nov = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 4 nov control/ResultsCorr.csv')
# times_4nov = [0, 4, 16, 29, 33, 39, 41, 45, 51, 54, 61, 69, 76, 84]
# nov4_dims = (352,342)
# nov4_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 4 nov control/ResultsACback.csv') 
# nov4_bckg = nov4_bckg['Mean'].to_numpy().tolist()


# times_4nov2corr = [t+10 for t in times_4nov2corr]

# ROIAC_6nov = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 6 nov control/ResultsCorr.csv')
# times_6nov = [0,9,16,23,26,30,37,43,45,61,70,74,77,81,88,96,99,103,111,118,123,129,133,141,149,154,161,168,174,181]
# nov6_dims = (329,316)
# nov6_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 6 nov control/ResultsACback.csv')
# nov6_bckg = nov6_bckg['Mean'].to_numpy().tolist()



# ROIAC_12nov = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/12 nov antibiotic/ResultsCorr.csv')

# times_12nov = [0,1,5,6,7,13,16,19,25,26,31,36,38,39,42,45,47,51,55,56,65,66,67,68,70,72,78,79,81,83,88,93,96,98,100,103,106,107,112,114,118,121,122,125,131,136,143,152,162]

# nov12_dims = (372,376)
# nov12_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/12 nov antibiotic/ResultsACback.csv')
# nov12_bckg = nov12_bckg['Mean'].to_numpy().tolist()




# ROIAC_5nov = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 5 nov antibiotic/ResultsCorr.csv')
# times_5nov = [0,2,10,14,17,23,29,34,38,42,47,52,56,62,69,75,79,85,88,93,95,104,114]
# nov5_dims = (329,316)
# nov5_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 5 nov antibiotic/ResultsACback.csv')
# nov5_bckg = nov5_bckg['Mean'].to_numpy().tolist()

# ROIAC_6dec = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 6 dec control/ResultsCorr.csv')
# times_6dec = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93]
# dec6_dims = (372,376)

# dec6_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/imagini 6 dec control/ResultsACback.csv')
# dec6_bckg = dec6_bckg['Mean'].to_numpy().tolist()

# ROIAC_16ian = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/16 ian CTRL/ResultsCorr.csv')
# ian16_dims = (372,376)
# times_16ian = [0, 3, 7, 9, 12, 16, 19, 21, 25, 28, 32, 36, 44, 48, 54, 57, 60, 65, 67, 69, 72, 75, 76, 78, 81, 84, 86, 87, 92, 94, 98, 98, 101, 108, 111, 112, 116, 119, 122, 126, 128]
# ian16_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/16 ian CTRL/ResultsACback.csv')
# ian16_bckg = ian16_bckg['Mean'].to_numpy().tolist()

# ROIAC_17ian = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/17 ian 2v5 CST/ResultsCorr.csv')
# ian17_dims = (372,376)
# times_17ian = [0, 6, 9, 12, 19, 23, 25, 27, 28, 32, 34, 37, 43, 44, 51, 55, 55, 58, 59, 63, 66, 68, 70, 73, 75, 77, 79, 81, 84, 87, 92, 93, 96, 98, 101, 105, 108, 111, 114, 115, 117, 120, 121, 123, 124, 127, 130, 132, 141, 158]
# ian17_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/17 ian 2v5 CST/ResultsACback.csv')
# ian17_bckg = ian17_bckg['Mean'].to_numpy().tolist()

# ROIAC_10dec = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/10 dec1mgL CST/ResultsCorr.csv')
# dec10_dims = (372,376)
# times_10dec = [0, 3,6,9,12,15,16,18,21,24,27,30,33,36,39,42,52,59,62,65,68,71,74,77,80,83,86,89,92,95,98,101,104,107,112,115,118,125,128,131,134,137,140,143,146,149,152,155,158,161,164,167,170,173,176,179,182,184]
# dec10_bckg = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/10 dec1mgL CST/ResultsACback.csv')
# dec10_bckg = dec10_bckg['Mean'].to_numpy().tolist()

# ROIAC_18dec = pd.read_csv('/Users/victorionescu/Desktop/imagini si rezultate/18 dec antibiotic/ResultsCorr.csv')
# dec18_dims = (372,376)
# times_18dec = [0, 3, 6, 10, 13, 16, 19, 22, 25, 28, 30, 36, 37, 49, 57, 60, 61, 62, 65, 66, 69, 72, 73, 75, 78, 81, 82, 84, 85, 88, 89, 90, 92, 93, 96, 101, 103, 104, 106, 109, 112, 115, 118, 119, 134]
# dec18_bckg = [1 for x in times_18dec]

#%% Surface class
class Surface:
    
    def __init__(self,name,cell,times,histogram,mx,mn,area,mean):
        self.name = name
        self.cell = cell
        self.times = times
        if histogram is None:
            histogram = [[0] for t in times]
        self.histogram = {t:h for t,h in zip(self.times,histogram)}
        self.min = {t:m for t,m in zip(self.times,mn)}
        self.max = {t:m for t,m in zip(self.times,mx)}
        self.area = {t:m for t,m in zip(self.times,area)}
        self.mean = {t:m for t,m in zip(self.times,mean)}
        self.histstart = {t:0 for t in self.times}
        self.histend = {t:65000 for t in self.times}
        self.figurebins()
        self.addmass()
        
    def addmajorar(self,majordict):
        m = deepcopy(majordict)
        maj = {}
        mino = {}
        ar = {}
        for t in self.times:
            maj.update({t:0.8*m[t]})
            minor = 4*self.area[t]/(np.pi*0.8*m[t])
            minor.update({t:minor})
            aspect = 0.8*m[t]/minor
            ar.update({t:aspect})
        self.major = deepcopy(maj)
        self.minor = deepcopy(mino)
        self.ar = deepcopy(ar)
        self.setslopes()
    def slopeandR(self,x, y):

        slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
        return slope,r_value**2
    def setslopes(self):
        dt = self.times[-1]-self.times[0]
        if dt == 0:
            self.areaslope = None
            self.majorslope = None
            self.arslope = None
            return
        areas = [self.area[t] for t in self.times]
        majors = [self.major[t] for t in self.times]
        ars = [self.ar[t] for t in self.times]
        tim = [t-self.times[0] for t in self.times]
        self.areaslope,self.areaslopeR2 = self.slopeandR(tim,areas)
        self.majorslope,self.majorslopeR2 = self.slopeandR(tim,majors)
        self.arslope,self.arslopeR2 = self.slopeandR(tim,ars)
      
        
    def addmass(self):
        mass = {}
        for t in self.times:
            m = np.sum(np.multiply(self.bins[t],self.histogram[t]))
            mass.update({t:m})
        self.mass = deepcopy(mass)
    def figurebins(self):
        l = len(self.histogram[self.times[0]])+1
        bins = {}
        for t in self.times:
            bins.update({t:np.linspace(self.histstart[t],self.histend[t],l)[:-1]})
        self.bins = deepcopy(bins)
    def scalehist(self):
        for t in self.times:
            self.histogram[t] = [x/self.area[t] for x in self.histogram[t]]
    def moments(self,x,y,index = 0):
        mean = np.sum((x)*y)
        std = np.sqrt(np.sum((x-mean)**2*y))
        skew = np.sum([(x-mean)**3*y])/(std**3)
        kurt = np.sum([(x-mean)**4*y])/(std**4)
        momlist = [mean,std,skew,kurt]
        return momlist[index]
    def addmomentsintime(self):
        # if (np.max(self.bins[self.times[0]])) > 100:
        #     print("Bins not normalized!")
        #     return
        self.hmean = deepcopy(self.mean)
        self.hstd = deepcopy(self.mean)
        self.hskew = deepcopy(self.mean)
        self.hkurt = deepcopy(self.mean)
        for t in self.times:
            
            self.hmean[t] = self.moments(self.bins[t],self.histogram[t],index = 0)
            self.hstd[t] = self.moments(self.bins[t],self.histogram[t],index = 1)
            self.hskew[t] = self.moments(self.bins[t],self.histogram[t],index = 2)
            self.hkurt[t] = self.moments(self.bins[t],self.histogram[t],index = 3)
            
            
            
            
            
#%% Cell Class         
class cell:
    
    def __init__(self,name,times,area,xpos,ypos,mean,median,mode,std,mn,mx,ellipse_angle,ellipse_major,ellipse_minor,skew,kurt,polesx,polesy,polesp,polessize,c_hist,c_max,c_min,c_count,c_mean,i_hist,i_max,i_min,i_count,i_mean,hist):
        ### Absolute measures; taken from dataframe (df) or inferred from own data ###
        self.name = name # Cell name ('Track_xxx') (df)
        self.duration = 0 # Cell duration; initially lasttime-initialtime, corrected by correctduration in experiment class (df, own measure later)
        self.times = times # Times at which cell is visible (df)
        self.area = {x:y for x,y in zip(times,area)} # Surface area(in px^2) of cell mask (df)
        self.position = {t:np.array([x,y]) for t,x,y in zip(times,xpos,ypos)} # XY position of mask/ROI centroid (df)
        self.mean =  {x:y for x,y in zip(times,mean)} # Mean intensity inside ROI (df)
        self.median = {x:y for x,y in zip(times,median)} # Median intensity inside ROI (df)
        self.mode = {x:y for x,y in zip(times,mode)} # Mode intensity inside ROI (df)
        self.std =  {x:y for x,y in zip(times,std)} # Standard dev of intensity inside ROI (df)
        self.min = {x:y for x,y in zip(times,mn)}
        self.max = {x:y for x,y in zip(times,mx)}
        self.angle =  {x:y for x,y in zip(times,ellipse_angle)} # Fit ellipse angle (df)
        self.major =  {x:y for x,y in zip(times,ellipse_major)} # Fit ellipse major (df)
        self.minor =  {x:y for x,y in zip(times,ellipse_minor)} # Fit ellipse minor (df)
        self.ar = {x:(self.major[x]/self.minor[x]) for x in times}
        if skew != None:
            self.skew =  {x:y for x,y in zip(times,skew)} # Histogram skweness (if present) (df)
        if kurt != None:
            self.kurt =  {x:y for x,y in zip(times,kurt)} # Histogram kurtosis (if present) (df)
        if '.' in name:
            self.parent_name = name[:-1] # Parent name based on cell name (inferred)
            if self.parent_name[-1] == '.':
                self.parent_name = self.parent_name[:-1]
        else:
            self.parent_name = None # No parents if name is not formatted like daughter cell
        self.polepositions = {t:np.array([np.asarray(x),np.asarray(y)]).T for t,x,y in zip(times,polesx,polesy)} # Pole positions from df
        self.polesprom = {x:y for x,y in zip(times,polesp)} # Pole prominence from df
        self.polessizes = {x:y for x,y in zip(times,polessize)} # Pole sizes from df
        self.setpolesdistances()
        ### Relative measures; given value when an experiment is initialized ###
        self.children = [] # Initialize children (set in experiment)
        self.children_name_list = [] # Initialize children names(set in experiment)
        self.parent= None # Initialize parent (set in experiment)
        self.colors = None # Initialize colors (set in experiment)
        self.provisionalparent = False # Alters plotting behaviour (set by self-plotting functions)
        self.density = {t:0 for t in times} # Initialize density (set in experiment)
        self.outlier = False # set outlier status to false initially; change if cell is abnormal
        self.contour = Surface(self.name,cell,times,c_hist,c_max,c_min,c_count,c_mean)
        self.interior = Surface(self.name,cell,times,i_hist,i_max,i_min,i_count,i_mean)
        self.histogram = {t:h for t,h in zip(self.times,hist)}
        self.histstart = {t:0 for t in self.times}
        self.histend = {t:65000 for t in self.times}
        self.figurebins()
        self.addmass()
        #self.interior.addmajorar(self.major)
        self.setslopes()
    ### MEMBER FUNTIONS ###
    
    def histdifference(self ,referencet = None):
        if referencet is None:
            print("Reference time not in cell; continuing with initial cell time as reference")
            referencet = self.times[0]
        refindex = self.times.index(referencet)
        initialhist = self.histogram[referencet]
        self.histdiff = {t:0 for t in self.times[refindex:]}
        self.histdiffintegral = {t:0 for t in self.times[refindex:]}
        for t in self.times[refindex:]:
            diff = np.asarray(self.histogram[t])-np.asarray(initialhist)
            self.histdiff[t] = diff.tolist()
            diff = np.abs(diff)
            diff = np.sum(diff)*500 # binsize, important
            self.histdiffintegral[t] = diff
            
    def addcorrelation(self,meancorr,mincorr,maxcorr,corrhists):
        self.correlation = Surface(self.name+'-corr',self,self.times.copy(),corrhists,maxcorr,mincorr,deepcopy(self.area),meancorr)
    def setpolesdistances(self):
        self.maxpoledistance = deepcopy(self.mean)
        self.expoleindexes = deepcopy(self.mean)
        for t in self.times:
            distmatrix = cdist(self.polepositions[t],self.polepositions[t],metric = 'euclidean')
            self.maxpoledistance[t] = np.max(distmatrix)
            initialexpole = np.unravel_index(np.argmax(distmatrix, axis=None), distmatrix.shape)
            pole1x = self.polepositions[t][initialexpole[0]][0]
            pole2x = self.polepositions[t][initialexpole[1]][0]
            if pole2x < pole1x:
                temp1 = initialexpole[1]
                temp2 = initialexpole[0]
                initialexpole = (temp1,temp2)
            self.expoleindexes[t] = initialexpole
                
    def normpolesprominence(self,bg):
        bdict = deepcopy(bg)
        for t in self.times:
            self.polesprom[t] = [x/self.interior.mean[t] for x in self.polesprom[t]]
            
        
    def histoverlap(self):
        if 'interior' not in self.__dict__.keys():
            print("Surfaces not set!")
            return
        overlapintcont = {}
        overlapintcell = {}
        overlapcontcell = {}
        for t in self.times:
            
            overlap = np.sum([min(els) for els in zip(self.histogram[t],self.interior.histogram[t])])/self.area[t]
            overlapintcell.update({t:overlap})
            overlap =np.sum([min(els) for els in zip(self.histogram[t],self.contour.histogram[t])])/self.area[t]
            overlapcontcell.update({t:overlap})
            overlap =np.sum([min(els) for els in zip(self.interior.histogram[t],self.contour.histogram[t])])/self.area[t]
            overlapintcont.update({t:overlap})
        self.intcontoverlap = deepcopy(overlapintcont)
        self.intcelloverlap = deepcopy(overlapintcell)
        self.contcelloverlap = deepcopy(overlapcontcell)
    
    def slopeandR(self,x, y):

        slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
        return slope,r_value**2
    
    def setslopes(self):
        dt = self.times[-1]-self.times[0]
        if dt == 0:
            self.areaslope = None
            self.majorslope = None
            self.arslope = None
            return
        areas = [self.area[t] for t in self.times]
        majors = [self.major[t] for t in self.times]
        ars = [self.ar[t] for t in self.times]
        tim = [t-self.times[0] for t in self.times]
        self.areaslope,self.areaslopeR2 = self.slopeandR(tim,areas)
        self.majorslope,self.majorslopeR2 = self.slopeandR(tim,majors)
        self.arslope,self.arslopeR2 = self.slopeandR(tim,ars)
        
    def addmass(self):
        mass = {}
        for t in self.times:
            m = np.sum(np.multiply(self.bins[t],self.histogram[t]))
            mass.update({t:m})
        self.mass = deepcopy(mass)
    
    def addACDCmeasures(self):
        if 'interior' not in self.__dict__.keys():
            print("No interior added!")
            return
        self.ACsolidity = deepcopy(self.mean)
        self.ACDCratio = deepcopy(self.mean)
        self.massratio = deepcopy(self.mean)
        for t in self.times:
            self.ACsolidity[t] = self.interior.area[t]/self.area[t]
            self.ACDCratio[t] = self.interior.hmean[t]/self.hmean[t]
            self.massratio[t] = self.interior.mass[t]/self.mass[t]
    
    def moments(self,x,y,index = 0):
        mean = np.sum((x)*y)
        std = np.sqrt(np.sum((x-mean)**2*y))
        skew = np.sum([(x-mean)**3*y])/(std**3)
        kurt = np.sum([(x-mean)**4*y])/(std**4)
        momlist = [mean,std,skew,kurt]
        return momlist[index]
    def addmomentsintime(self):
        # if (np.max(self.bins[self.times[0]])) > 100:
        #     print("Bins not normalized!")
        #     return
        self.hmean = deepcopy(self.mean)
        self.hstd = deepcopy(self.mean)
        self.hskew = deepcopy(self.mean)
        self.hkurt = deepcopy(self.mean)
        for t in self.times:
            self.hmean[t] = self.moments(self.bins[t],self.histogram[t],index = 0)
            self.hstd[t] = self.moments(self.bins[t],self.histogram[t],index = 1)
            self.hskew[t] = self.moments(self.bins[t],self.histogram[t],index = 2)
            self.hkurt[t] = self.moments(self.bins[t],self.histogram[t],index = 3)
    

    def figurebins(self):
        l = len(self.histogram[self.times[0]])+1
        bins = {}
        for t in self.times:
            bins.update({t:np.linspace(self.histstart[t],self.histend[t],l)[:-1]})
        self.bins = deepcopy(bins)
    def scalehist(self):
        for t in self.times:
            self.histogram[t] = [x/self.area[t] for x in self.histogram[t]]
    
    # Flips y axis for 0 to be at x = 0
    def flipycoords(self,ysize):
        for t in self.times:
            self.position[t][1] = ysize - self.position[t][1]
            
    # Create TreeNode object out of self and store node  
    def makenodes(self):
        node = TreeNode(self)
        self.node = node
        for c in self.children:
            c.makenodes()
            node.add_child(c.node)
        self.node = node
    # Update color only based on DENSITY; for other uses self.color works
    def updatecolor(self):
        if 'density' not in self.__dict__.keys():
            print("Density has not been set!")
            return
        self.color = {t:self.density[t] for t in self.times}
        return
    
    
    # Helper function of changepolecoords;compute a rotation matrix to rotate pole vectors to (1,0)
    def getcrotationmatrix(self,angle):
        angle = -angle
        angle = np.radians(angle)
        ang = angle
        mat = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        return mat
    
    # Function upon initializing an experiment; finds cells maching name in passed cellist (usually from experiment class)
    def findchildren(self,cell_list):
        for i in cell_list:
            if i.name[:-1] == self.name:
                self.children.append(i)
                self.children_name_list.append(i.name)
            if i.name[:-2] == self.name and '.' not in self.name:
                self.children.append(i)
                self.children_name_list.append(i.name)
            if i.name == self.parent_name:
                self.parent = i
    
    # Shift (to (0,0)), rotate (to (1,0) vector) and scale (optional) pole coords to overlap on a generic cell
    def changepolecoords(self,image_size,scaled = False):
        for t in self.times:
            x = self.position[t][0]
            y = image_size[1] -self.position[t][1]
            self.polepositions[t][:,1] =  image_size[1]-self.polepositions[t][:,1]
            self.polepositions[t] = self.polepositions[t] - np.array([x,y])
            a = self.angle[t]
            self.polepositions[t] = np.matmul(self.polepositions[t],self.getcrotationmatrix(a).T)
            # if scaled:
            #     vec = np.array([[self.major[t],2*self.minor[t]]]).T
            #     self.polepositions[t] = self.polepositions[t]/vec
    
     # Helper function for getting min max values from attribute dictionary          
    def getminmax(self,value_name):
        chosen_value = deepcopy(self.__dict__[value_name])
        
        mn = chosen_value[self.times[0]]
        mx = chosen_value[self.times[0]]
        for t in chosen_value.keys():
            if chosen_value[t] != None:
                if chosen_value[t] < mn:
                    mn = chosen_value[t]
                if chosen_value[t]> mx:
                    mx = chosen_value[t]
        
        return mn,mx 
    
    # Helper function to normalize (minmax norm) attribute dictionary
    def normalizevalue(self,value_name, mn,mx):
        value = deepcopy(self.__dict__[value_name])
        neg_values = False
        if mn< 0:
            neg_values = True
            mn = abs(mn)
        for i in value.keys():
            if neg_values:
                value[i] = value[i]+mn
            value[i] = (value[i]-mn)/(mx-mn)
        newname = value_name+'norm'
        self.__dict__.update({newname:value})
    
    # Same as getminmax, but works on dict of lists
    def getminmaxpoles(self):
        mn =self.polesprom[self.times[0]][0]
        mx = self.polesprom[self.times[0]][0]
        for t in self.polesprom.keys():
            if self.polesprom[t] != [None]:
                if min(self.polesprom[t]) < mn:
                    mn = min(self.polesprom[t])
                if max(self.polesprom[t])> mx:
                    mx = max(self.polesprom[t])
        return mn,mx
    
    # Same as normattr, but works on dict of lists
    def normalizepoles(self, mn,mx):
        value = deepcopy(self.polesprom)
        neg_values = False
        if mn < 0:
            neg_values = True
            mn = abs(mn)
        for i in value.keys():
            if neg_values:
                value[i] = [x+mn for x in value[i]]
            value[i] = [(x-mn)/(mx-mn) for x in value[i]]
        newname = 'polespromnorm'
        self.__dict__.update({newname:value})
        
    # Main plotting function for individual cell attributes
    # Can extend to connect cell lineages, or return processed values without plotting
    def plotvalue(self,value_name,t_shift = 0,plotflag = True,addouts = False):
        
        valuedic = self.__dict__[value_name]
        if self.parent == None or self.provisionalparent == True:
            yvalues = [valuedic[t] for t in self.times]  
            xvalues = self.times
            xvalues = [x-t_shift for x in xvalues]
            self.provisionalparent = False
            colorlist =[plt.cm.magma(self.colornorm[t]) for t in self.times]
            colorlist = np.array([np.asarray(el) for el in colorlist])
        else:
            par_time = self.parent.times[-1]
            par_value = self.parent.__dict__[value_name][par_time]
            yvalues = [par_value]+[valuedic[t] for t in self.times]
            xvalues = [par_time] + self.times
            xvalues = [x-t_shift for x in xvalues]
            colorlist =[plt.cm.magma(self.colornorm[t]) for t in self.times]
            colorlist = [plt.cm.magma(self.parent.colornorm[par_time])]+colorlist
            colorlist = np.array([np.asarray(el) for el in colorlist])
            
        
        if plotflag and (not self.outlier or addouts):
            if type(self.color) == str:
                
                plt.plot(xvalues,yvalues,c = self.color,alpha = 0.5)
                plt.scatter(xvalues,yvalues,c = self.color,alpha = 0.5)
            elif type(self.color) == float:
    
                plt.plot(xvalues,yvalues,c = plt.cm.viridis(self.color),alpha = 0.1)
                plt.scatter(xvalues,yvalues,c = plt.cm.viridis(self.color),alpha = 0.3)
            else:
                
                plt.plot(xvalues,yvalues,c = plt.cm.magma(colorlist[0,0]),alpha = 0.9,zorder = 0)
                plt.scatter(xvalues,yvalues,c = colorlist,cmap = 'magma',norm = 'linear',alpha = 1,zorder = 5)
            
        
        else:
            
            d = self.times[-1]-self.times[0]
            duration = [d for x in xvalues]
            
            return xvalues,yvalues,duration,colorlist
        
    
    # Runs on experiment startup, computes self solidity based on std:mean ratio of intensity normal distribution
    def setsolidity(self,th):
        soldic = {t:0 for t in self.times}
        cvdic = {t:0 for t in self.times}
        means = self.mean
        std = self.std
        threshold1 = {t:means[t]*0.3 for t in self.times}
        threshold2 = {t:means[t]*1.3 for t in self.times}
        
        for t in self.times:
            z1 = ((threshold1[t]-self.mean[t])/self.std[t])
            z2 = ((threshold2[t]-self.mean[t])/self.std[t])
            soldic[t] = st.norm.cdf(z2)-st.norm.cdf(z1)
            cvdic[t] = std[t]/means[t]
        newname = 'solidity'
        newname2 = 'cv'
        self.__dict__.update({newname:soldic})
        self.__dict__.update({newname2:cvdic})
    
    # Almost useless, can just update attribute directly
    def makeprovparent(self):
        self.provisionalparent = True
        return self
    
    # Attempt to correct cycle duration based on interval between frames and standard duration
    # Works on cells with 1.) short base durations 2.) long intervals before and after to steal time
    # Computes what fraction of interval should come out of which (based on avg cycle duration set to 18)
    def correctduration(self):
        duration = self.times[-1]-self.times[0]
        self.duration = duration
        if self.parent == None:
            return
        preinterval = self.times[0]-self.parent.times[-1]
        postinterval = 0
        if self.children != []:
            postinterval = self.children[0].times[0]-self.times[-1]
        
        if duration < 10:
            if postinterval <9:
                postinterval = 0
            if preinterval <9:
                preinterval = 0
            if duration == 0:
                prescalefactor = 0.5
                postscalefactor = 0.5
            else:
                if preinterval != 0:
                    prescalefactor =  (18 - duration)/preinterval
                else:
                    prescalefactor = 0
                if postinterval != 0:
                    postscalefactor = (18-duration)/postinterval
                else:
                    postscalefactor = 0
            self.duration += (preinterval*prescalefactor +postinterval*postscalefactor)/2
            
    def finddominantfrequency(self,value_name):
        if value_name not in self.__dict__.keys():
            print("Value not in cell!")
            return
        x = self.times
        x = [t-self.times[0] for t in self.times]
        factor= x[-1]-x[0]
        if factor != 0:
            factor = 18/factor
        x = [t*factor for t in x]
        y = np.array([self.__dict__[value_name][t] for t in self.times])
        y = (y - np.mean(y))
        duration = self.times[-1] - self.times[0]
        if duration == 0:
            self.dfreq = np.array([None])
            self.amp = np.array( [None])
            return
        duration = 18
        dmax = np.max( np.diff(self.times))
        n = len(self.times)
        freqs = np.linspace(1/duration, n/duration, 15*n)*2*np.pi
        periodogram = lombscargle(x, y, freqs)
        periodogram = np.array(periodogram)
        freqs = np.array(freqs)
        
        maximalist = argrelextrema(periodogram,np.greater)
        if maximalist[0].size ==0:
            self.dfreq = np.array([None])
            self.amp = np.array( [None])
            return
       
        self.dfreq = np.array([freqs[i] for i in maximalist[0]])
        self.amp = np.array([np.sqrt(4*periodogram[i]/(5*n)) for i in maximalist[0]])
        self.periodogram = (freqs,periodogram)
        
        
    def poleFFT(self,sampt = 20):
        polevalues = []
        for t in self.times:
            expoles = [self.polesprom[t][i] for i in self.expoleindexes[t]]
            polevalues.append(np.mean(expoles))
        
        polevalues = [x-np.mean(polevalues) for x in polevalues]
        duration = (self.times[-1]-self.times[0])*60
        if duration == 0:
            print("Cell seen for 0 minutes!")
            self.expolefourier = ([None],[None])
            return
        
        samplingtime = sampt# in sec
        N = int(duration/samplingtime)
        time = np.arange(0,N*samplingtime,samplingtime)
        try:
            spl = interpolate.PchipInterpolator([(t-self.times[0])*60 for t in self.times], polevalues)
        except ValueError:
            print(self.name)
            print(polevalues)
            return
        signal = spl(time)
        
        w = hamming(N)
        polefft = fft(signal,norm = 'forward')
        polefft = np.abs(polefft[0:N//2])
        polefreqs = fftfreq(N, d = samplingtime)[0:N//2]
        self.polesignal = (time,signal)
        self.expolefourier = (polefreqs,polefft)
    
    def addbackgroundratio(self,bgdict):
        bdict = deepcopy(bgdict)
        self.ratio = deepcopy(self.mean)
        for t in self.times:
            if t in bgdict.keys():
                self.ratio[t] /= bgdict[t]
                # self.interior.histstart[t] /= bdict[t]
                # self.interior.histend[t] /= bdict[t]
                # self.interior.figurebins()
                # self.contour.histstart[t] /= bdict[t]
                # self.contour.histend[t] /= bdict[t]
                # self.contour.figurebins()
                # self.histstart[t] /= bdict[t]
                # self.histend[t] /= bdict[t]
                # self.figurebins()
        print("Added background ratio to cells!")
    def addsolbckgd(self):
        if 'ratio' in self.__dict__.keys():
            self.solb = deepcopy(self.solidity)
            for t in self.times:
                self.solb[t] = self.ratio[t]*self.solb[t]
        else:
            print("No background!")
            return
    def scalemeans(self,bg,bs):
        bgdict = deepcopy(bg)
        stddict = deepcopy(bs)
        for t in self.times:
            if t in bgdict.keys():
                self.mean[t] /= bgdict[t]
                self.std[t] /= stddict[t]
                
    def setsolidareaself(self,lower = 0.3,upper = 1.3):
        self.solsel = deepcopy(self.area)
        if 'histogram' not in self.__dict__.keys():
            print("Do not have histogram")
            return
        for t in self.times:
            val = [x for x in self.bins[t] if x >= lower*self.ratio[t] and x <= upper*self.ratio[t]]
            if val == []:
                self.solsel[t] = 0
                continue
            thindex = np.where(self.bins[t] == val[0])[0][0]
            thindex2 = np.where(self.bins[t] == val[-1])[0][0]
            self.solsel[t] = np.sum(self.histogram[t][thindex:thindex2])
    def setsolidarea(self,lower = 0.2,upper = 0.7):
        self.solarea = deepcopy(self.area)
        if 'histogram' not in self.__dict__.keys():
            print("Do not have histogram")
            return
        for t in self.times:
            val = [x for x in self.bins[t] if x > 1 and x > lower*(np.max(self.bins[t])-1) and x < upper*(np.max(self.bins[t])-1)]
            if val == []:
                self.solarea[t] = 0
                continue
            thindex = np.where(self.bins[t] == val[0])[0][0]
            thindex2 = np.where(self.bins[t] == val[-1])[0][0]
            self.solarea[t] = np.sum(self.histogram[t][thindex:thindex2])
             
    def sethollowarea(self,lower = 0,upper = 1):
        self.holarea = deepcopy(self.area)
        if 'histogram' not in self.__dict__.keys():
            print("Do not have histogram")
            return
        for t in self.times:
            val = [x for x in self.bins[t] if x > 1 and x > lower*(np.max(self.bins[t])-1) and x < upper*(np.max(self.bins[t])-1)]
            if val == []:
                self.holarea[t] = 0
                continue
            thindex = np.where(self.bins[t] == val[0])[0][0]
            thindex2 = np.where(self.bins[t] == val[-1])[0][0]
            self.holarea[t] = np.sum(self.histogram[t][thindex:thindex2])
            
            
            
            
            
            
#%% Experiment class
class experiment:
    
    def __init__(self,name,image_size,times,cells,antibiotic,backgrounds = None,backgroundstds = None):
        
        self.name = name # should include date and antibiotic/control, maybe concentration
        self.image_size = image_size # Tuple that gives size of series images
        self.times = times # Experimetn global times, passed as list
        self.cells = cells # list of cell objects
        if backgrounds != None:
            self.backgrounds = {t:b for t,b in zip(self.times,backgrounds)}
        else:
            self.backgrounds = None
        if backgroundstds != None:
            self.bstds = {t:b for t,b in zip(self.times,backgroundstds)}
        else:
            self.bstds = None
        # Initialize relative values for cells in list
        for i in self.cells:
            i.findchildren(self.cells)
            i.changepolecoords(image_size,scaled = True)
            i.flipycoords(image_size[1])
            i.correctduration()
            if self.backgrounds != None:
                b = self.backgrounds
                i.normpolesprominence(b)
                i.histoverlap()
                i.scalehist()
                i.interior.scalehist()
                i.contour.scalehist()
                i.addbackgroundratio(b)
                i.setsolidareaself()
                i.setsolidarea()
                i.sethollowarea()
                i.addmomentsintime()
                i.interior.addmomentsintime()
                i.contour.addmomentsintime()
                i.addACDCmeasures()
                i.poleFFT()
            i.histdifference()
                
                # i.scalemeans(b,bs)
        self.polmin,self.polmax = self.normpoles() #minmax pole prominences (?)
        self.antibiotic = antibiotic # Antibiotic time
        self.getcellcount() # make new dict that contains cellcounts for self.times
        rootcells = [cell for cell in self.cells if '.' not in cell.name] # (For fully tracked experiments): Stores original cell nodes in list
        # Make lineage trees starting at root cells, while also setting generation in TreeNode objects (useful later)
        self.lineagetrees = []
        self.lineagetrees = self.makelineagetrees(rootcells)
        self.getdensity() #compute and set distance from COM for all cells
        self.normalizecellattr('color') # set color based on density
        self.computemedianduration()
        self.setoutliers()
        
    def resetselsol(self,lower,upper):
        for c in self.cells:
            c.setsolidareaself(lower,upper)
    def resetsolarea(self,lower,upper):
        for c in self.cells:
            c.setsolidarea(lower,upper)
            
    def resetholarea(self,lower,upper):
        for c in self.cells:
            c.sethollowarea(lower,upper)
                
    # Debugging code, can remove
    def printlineagedata(self):
        for c in self.lineagetrees:
            c.pre_order_traversal()
    
    # Traverse trees from root nodes and make them nodes themselves
    def makelineagetrees(self,rootcells):
        for c in rootcells:
            c.makenodes()
        lintrees = [c.node for c in rootcells]
        for c in lintrees:
            c.setgenerations()
        return lintrees
    
    # Get global (per timepoint) min and max and normalize all cells according to that (minmax norm)
    def normalizecellattr(self,cellvalue):
        minlist = []
        maxlist = []
        for i in self.cells:
            mn,mx = i.getminmax(cellvalue)
            minlist.append(mn)
            maxlist.append(mx)
        minimum = min(minlist)
        maximum = max(maxlist)
        for i in self.cells:
            i.normalizevalue(cellvalue, minimum, maximum)
        print("Created new cell attribute with name: "+cellvalue+"norm")
    
    # useless
    def getinternalsolthreshold(self):
        self.averagevalue('median')
        th = 0
        return th
        
        
        
    
    def normpoles(self):
        minlist = []
        maxlist = []
        for i in self.cells:
            mn,mx = i.getminmaxpoles()
            minlist.append(mn)
            maxlist.append(mx)
        minimum = min(minlist)
        maximum = max(maxlist)
        for i in self.cells:
            i.normalizepoles(minimum, maximum)
        print("Created new cell attribute (dict of lists) with name: polespromnorm")
        return minimum,maximum
    
    def plotpoles(self):
        for t in self.times:
            for c in self.cells:
                if t in c.times and c.polesprom[t] != [None]:
                   
                    polesc = deepcopy(c.polespromnorm)
                    alpha = [x  for x in polesc[t]]
                    alpha = np.nan_to_num(alpha)
                    
                    plt.scatter(c.polepositions[t][:,0],c.polepositions[t][:,1],alpha = alpha, c = plt.cm.viridis(polesc[t]),s=np.nan_to_num(c.polessizes[t]))
            plt.xlim((-40,40))
            plt.ylim((-40,40))
            if t < self.antibiotic:
                add = ' pre-antibiotic: T = '
            else:
                add = ' post-antibiotic: T = '
            plt.title(self.name+ add+str(t)+' min.')
            
            plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(self.polmin, self.polmax), cmap='viridis'),
                       orientation='vertical', label= 'Prominence')
            
            plt.show()
            
            

    def getglobalattr(self,value_name):
        globalvaluedic = {t:[] for t in self.times}
        for t in self.times:
            for c in self.cells:
                if t in c.times:
                    x = c.__dict__[value_name][t]
                    globalvaluedic[t].append(x)
        self.__dict__.update({value_name:globalvaluedic})
    
    def getcellcount(self):
        countdic = {t:0 for t in self.times}
        for t in self.times:
            for c in self.cells:
                if t in c.times:
                    countdic[t] += 1
        self.__dict__.update({'cellcount':countdic})
    
    def setcmap(self,color = None):
    
        clist = [(x-self.times[0])/(self.times[-1]-self.times[0]) for x in self.times]
        cmap = {t:c for t,c in zip(self.times,clist)}
        for c in self.cells:
            c.color = cmap[c.times[0]]
        if color != None:
            for c in self.cells:
                c.color = color
    
    def getdensity(self):
        
        for t in self.times:
         
            clist = [c for c in self.cells if t in c.times]
            print(len(clist))
            positions = [c.position[t] for c in clist]
            positions = np.asarray(positions)
            if len(positions.shape) != 2: #cell is alone 
                for c in clist:
                    c.density[t] = 0
                    c.updatecolor()
                continue
            dists = cdist(positions,positions)
            for i in range(dists.shape[0]):
                d = dists[i,:]
                thresh = clist[i].major[t]
                d[d<thresh] = 0
                d = 1/d
                d = np.nan_to_num(d,posinf = 0)
                dists[i,:] = d
                
            
            
            dists = np.sum(dists,axis = 1)
            for i,c in enumerate(clist):
                c.density[t] = dists[i]
                c.updatecolor()
            
            
                    
    
        
    def plotlineage(self,cellist,value_name,t_shift = 0, batch = True, color = None):
        
        for c in cellist:
            
            c.plotvalue(value_name,t_shift)
        if not batch:
            plt.show()
    
    def plotseparate(self,value_name,cellist = None,t_shift = 0,batch = True,color = None):
        self.setcmap(color= color)
        for c in cellist:
            
            c.plotvalue(value_name,t_shift)
        if not batch:
            plt.show()
    def averagevalue(self,value_name):
        if value_name not in self.__dict__.keys():
            self.getglobalattr(value_name)
        avgdict = {t:0 for t in self.times}
        stddict = {t:0 for t in self.times}
        for t in self.times:
            avg = np.mean(self.__dict__[value_name][t])
            std = np.std(self.__dict__[value_name][t])
            avgdict[t] = avg
            stddict[t] = std
        avglist = [avgdict[t] for t in self.times]
        stdlist = [stddict[t] for t in self.times]
        avg_name = value_name+'_avg'
        std_name = value_name+'_std'
        self.__dict__.update({avg_name:avglist})
        self.__dict__.update({std_name:stdlist})
        
    def getlineagedics(self,cellist,branches = None):
        if cellist == []:
            return
        lineagedic = {n.name:[] for n in cellist}
        for k in lineagedic.keys():
            if branches == None:
                lineagedic[k] = [cell for cell in self.cells if k in cell.name]
            else:
                for b in branches:
                    lineagedic[k] = lineagedic[k] + [cell for cell in self.cells if b not in cell.name]
        return lineagedic
        
    
    def plotaverage(self,value_name,color = 'b',vshift = 0):
        name = value_name +'_avg'
        if name not in self.__dict__.keys():
            self.averagevalue(value_name)
        avgs = self.__dict__[name]
        stds = self.__dict__[value_name+'_std']
        markers, caps, bars = plt.errorbar([x-self.antibiotic for x in self.times],[x+vshift for x in avgs],yerr = stds, c= color,capsize = 3, label = self.name,fmt = 'o')
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]
        plt.scatter([x-self.antibiotic for x in self.times],[x+vshift for x in avgs], c= color)
        plt.plot([x-self.antibiotic for x in self.times],[x+vshift for x in avgs], c= color,alpha = 0.3)
        #plt.axvline(x = self.antibiotic,ymin = 0,ymax = 1,c = color)
        
    def addsolidity(self,threshold_time = None):
        if threshold_time == None:
            th = self.getinternalsolthreshold()
        else:
            self.averagevalue('mean')
            th = self.mean_avg[self.times.index(threshold_time)]
        for c in self.cells:
            c.setsolidity(th)
        print("Added solidity")
        
    def splitbyinterval(self,interval):
        pretime = [x for x in self.times if x < self.antibiotic]
        count = int(np.ceil(self.antibiotic/interval)+1)
        intervalsdict = {-i:[] for i in range(1,count)}
        for i in range(1,count):
            intervalsdict[-i] = [t for t in pretime if t > self.antibiotic - i*interval]
            pretime = [t for t in pretime if t not in intervalsdict[-i]]
        newname = 'intervals'
        self.__dict__.update({newname:intervalsdict})
        
    def plotinterval(self,interval_key,value,color = None):
        if 'intervals' not in self.__dict__.keys():
            print("Split times by interval first!")
            return
        if interval_key not in self.intervals:
            return
        cellist = []
        for t in self.intervals[interval_key]:
            temp = [c.makeprovparent() for c in self.cells if c.times[0] == t]
            
            cellist = cellist+temp
        lineagedic = self.getlineagedics(cellist)
        if lineagedic == None:
            return
        for k in lineagedic.keys():
            self.plotlineage(lineagedic[k],value,self.antibiotic,batch = True,color = color)
        
    def plotcells(self,clist = None):
        if clist != None:
            for t in self.times:
                ells = []
                for c in clist:
                    if t in c.times:
                        x = c.position[t][0]
                        y = c.position[t][1]
                        w = c.major[t]
                        h = c.minor[t]
                        ang = c.angle[t]
                        
                        ells.append(Ellipse(xy=(x,y) ,
                            width=w, height=h,
                            angle=ang, fc = plt.cm.magma(c.colornorm[t])))
                fig, ax = plt.subplots()
                ax.set_xlim(0,self.image_size[0])
                ax.set_ylim(0,self.image_size[1])
                ax.set_title("T="+str(t))
                if ells != []:
                    for e in ells:
                        ax.add_artist(e)
                    fig.set_title = ("T = "+str(t))
                    # fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(mn, mx), cmap='viridis'),
                    #           ax=ax, orientation='vertical', label= '')
                    plt.show()
        else:
            for t in self.times:
                ells = []
                for c in self.cells:
                    if t in c.times:
                        x = c.position[t][0]
                        y = c.position[t][1]
                        w = c.major[t]
                        h = c.minor[t]
                        ang = c.angle[t]
                        
                        ells.append(Ellipse(xy=(x,y) ,
                            width=w, height=h,
                            angle=ang, fc = plt.cm.magma(c.colornorm[t])))
                fig, ax = plt.subplots()
                ax.set_xlim(0,self.image_size[0])
                ax.set_ylim(0,self.image_size[1])
                ax.set_title("T="+str(t))
                if ells != []:
                    for e in ells:
                        ax.add_artist(e)
                    ax.set_aspect("equal")
                    fig.set_title = ("T = "+str(t))
                    # fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(mn, mx), cmap='viridis'),
                    #           ax=ax, orientation='vertical', label= '')
                    plt.show()
        
    def getfreqbygeneration(self,value_name,plot = False):
        roots = self.lineagetrees
        mx = 0
        for r in roots:
            depth = r.maxdepthdown()
            if depth>mx:
                mx = depth
        freqlist = []
        amplist = []
        for i in range(mx):
            freq = []
            amp = []
            for c in self.cells:
                if c.node.generation == i:
                    c.finddominantfrequency(value_name)
                    freq += list(c.dfreq)
                    amp += list(c.amp)
            freqlist.append(freq)
            amplist.append(amp)
            if plot:
                plt.scatter(freq,amp)
                plt.xlabel("Frequency (rad/s)")
                plt.ylabel("Amplitude (solidity units)")
                plt.title("Gen "+str(i))
                plt.show()
    
        
        return freqlist,amplist
    
    
    def computemedianduration(self):
        roots = self.lineagetrees
        mx = 0
        for r in roots:
            depth = r.maxdepthdown()
            if depth>mx:
                mx = depth
        self.genduration = {g:0 for g in range(mx)}
        for i in range(mx):
            median = []
            for c in self.cells:
                if c.node.generation == i:
                    d = c.times[-1]-c.times[0]
                    if d > 0:
                        median.append(d)
                        
            med = np.median(median)
            self.genduration.update({i:med})
            
            
    def setoutliers(self):
        for c in self.cells:
            major = [c.major[t] for t in c.times]
            majorprog = np.diff(major)
            checkneg = np.array([True if -x > 0.3*y else False for x,y in zip(majorprog,major)])
            if np.any(checkneg):
                c.outlier = True
            g = c.node.generation
            standard = self.genduration[g]
            d = c.times[-1]-c.times[0]
            
            if (0.4*standard >= d or 2.5*standard <= d) and d != 0:
                c.outlier = True
                               
#%% TreeNode class            
class TreeNode:
    def __init__(self, cell):
        self.cell = cell
        self.children = []
        
    def add_child(self, child):
        if type(child) == list:
            self.children = self.children + child
        else:
            self.children.append(child)
    

    # Create a function for depth-first traversal (pre-order)
    def pre_order_traversal(self,nbranches = None):
        if self is None:
            return
        print(self.cell.name)
        print("Ellangle: "+str(self.cell.angle))
        print("Ellmaj: "+str(self.cell.major))
        print("Ellmin: "+str(self.cell.minor))
        print("Positions: "+str(self.cell.position))
        counter = 0
        for child in self.children:
            
            child.pre_order_traversal()
            counter +=1
    def setgenerations(self,counter=0):
        if self is None:
            return
        self.generation = counter
        counter +=1
        for i in self.children:
            i.setgenerations(counter)
    def maxdepthdown(self):
        if self == None:
            return 0
        if self.children != []:
            leftdepth = self.children[0].maxdepthdown()
            rightdepth = self.children[-1].maxdepthdown()
        else:
            leftdepth = 0
            rightdepth = 0
        
        if leftdepth > rightdepth:
            return leftdepth+1
        else:
            return rightdepth+1
        


#%% Processing and importing from data frames

   


def getROIcells(dataframe,time):
    celllist = []
    df = dataframe.copy(deep = True)
    times = time.copy()
    needed_labels = ['ROI name','Area','X','Y','Mean','Median','Mode','StdDev','Min','Max','Angle',
                     'Major','Minor','Skew','Kurt',
                     'MAX X POS','MAX Y POS','MAX P', 'MAX SIZE',
                     'Contour Hist','Contour Max','Contour Min','Contour Count','Contour Mean','Interior Hist','Interior Max','Interior Min','Interior Count','Interior Mean','Cell Hist']
    dic_list = []
    for i in needed_labels:
        dic_list.append(getROIdata(df,i,times))
    
    for i in dic_list[0].keys():
        
        celllist.append(cell(dic_list[0][i][1][0],dic_list[0][i][0],dic_list[1][i][1],dic_list[2][i][1],
                             dic_list[3][i][1],dic_list[4][i][1],dic_list[5][i][1],dic_list[6][i][1],dic_list[7][i][1],
                             dic_list[8][i][1],dic_list[9][i][1],dic_list[10][i][1],dic_list[11][i][1],dic_list[12][i][1],
                             dic_list[13][i][1],dic_list[14][i][1],dic_list[15][i][1],dic_list[16][i][1],dic_list[17][i][1],
                             dic_list[18][i][1],dic_list[19][i][1],dic_list[20][i][1],dic_list[21][i][1],dic_list[22][i][1],
                             dic_list[23][i][1],dic_list[24][i][1],dic_list[25][i][1],dic_list[26][i][1],dic_list[27][i][1],dic_list[28][i][1],dic_list[29][i][1]))
    return celllist


# process Trackmate data
def processTM(dataframe, int_values,times):
    df = dataframe.copy(deep=True)
    df = df.drop(index = df.index[:3],axis = 0,inplace = False)
    df = df.reset_index()
    Frames = df["FRAME"].copy()
    Time_Series = Frames.copy()
    for i in range(len(Time_Series)):
        Time_Series[i] = str(times[int(Frames[i])])
    Time_Series = Time_Series.rename("TIMES")
    df = pd.concat([df,Time_Series],axis = 1)
    df = df.convert_dtypes()
    df[int_values] = df[int_values].astype(float)
    df['POSITION_T'] = df['POSITION_T'].astype(float)
    df['POSITION_T'] = df['POSITION_T'].astype(int)
    df = df. sort_values(by = "TIMES")
    return df


#process ROI data
def processROIs(dataframe,int_values,times,tracks = True):
    df = dataframe.copy(deep = True)
    Slices = df['Slice']
    Time_Series2 = Slices.copy()
    for i in range(len(Time_Series2)):
        Time_Series2[i] = str(times[int(Slices[i])-1])
    Time_Series2 = Time_Series2.rename("Times")
    df = pd.concat([df,Time_Series2],axis = 1)
    df['Times'] = df['Times'].astype(int)
    if 'MAX X POS' in df.columns:
        df['MAX X POS']  = df['MAX X POS'].astype(str)
        df['MAX X POS'] = df['MAX X POS'].apply(lambda x:x.split(','))
        df['MAX X POS'] = df['MAX X POS'].apply(lambda x:list(map(float,x)))
        df['MAX Y POS']  = df['MAX Y POS'].astype(str)
        df['MAX Y POS'] = df['MAX Y POS'].apply(lambda x:x.split(','))
        df['MAX Y POS'] = df['MAX Y POS'].apply(lambda x:list(map(float,x)))
        df['MAX P']  = df['MAX P'].astype(str)
        df['MAX P'] = df['MAX P'].apply(lambda x:x.split(','))
        df['MAX P'] = df['MAX P'].apply(lambda x:list(map(float,x)))
    if 'MAX SIZE' in df.columns:
        df['MAX SIZE']  = df['MAX SIZE'].astype(str)
        df['MAX SIZE'] = df['MAX SIZE'].apply(lambda x:x.split(','))
        df['MAX SIZE'] = df['MAX SIZE'].apply(lambda x:list(map(float,x)))
    if 'Contour Hist' in df.columns:
        df['Contour Hist'] = df['Contour Hist'].astype(str)
        df['Contour Hist'] = df['Contour Hist'].apply(lambda x:x.split(','))
        df['Contour Hist'] = df['Contour Hist'].apply(lambda x:list(map(float,x)))
    if 'Interior Hist' in df.columns:
        df['Interior Hist'] = df['Interior Hist'].astype(str)
        df['Interior Hist'] = df['Interior Hist'].apply(lambda x:x.split(','))
        df['Interior Hist'] = df['Interior Hist'].apply(lambda x:list(map(float,x)))
    if 'Cell Hist' in df.columns:
        df['Cell Hist'] = df['Cell Hist'].astype(str)
        df['Cell Hist'] = df['Cell Hist'].apply(lambda x:x.split(','))
        df['Cell Hist'] = df['Cell Hist'].apply(lambda x:list(map(float,x)))
    df = df.convert_dtypes()
    if tracks:
        df = df.drop(index = df.index[ df['ROI name'].str.contains('ID') ],axis = 0, inplace = False)
    return df




       
    

    
def plotTMdata(dataframe,value_name,times):
    names = dataframe['LABEL'].copy()
    unique_names = names.unique()
    branchtimes = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        branchtimes[i] = dataframe[dataframe["LABEL"] == i]["TIMES"].to_list()
    brancharray = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        brancharray[i] = dataframe[dataframe["LABEL"] == i][value_name].to_list()
    min_time = min(times)
    max_time = max(times)
    c_times = [(n-min_time)/(max_time-min_time) for n in times]
    colorbranchtimes = branchtimes.copy()
    for i in colorbranchtimes.keys():
        colorbranchtimes[i] = times.index(min(colorbranchtimes[i]))
    initial_keys = []
    for i in branchtimes.keys():
        if '.' not in i:
            initial_keys.append(i)

    for j in initial_keys:
        for i in branchtimes.keys():
            if j in i:
                if j != i:
                    parent_index = i[:-1]
                    if parent_index[-1] == '.':
                        parent_index = parent_index[:-1]
                    parent_endtime = branchtimes[parent_index][-1]
                    parent_endvalue = brancharray[parent_index][-1]
                    branchtimes[i] = [parent_endtime]+branchtimes[i]
                    brancharray[i] = [parent_endvalue]+brancharray[i]
                color = plt.cm.viridis(c_times[colorbranchtimes[i]])
               
                plt.plot(branchtimes[i],brancharray[i],c = color)
                plt.axvline(x = branchtimes[i][0],color = color,label = 'Divisions', ls = '--')
        
        plt.title("Overall "+value_name)
        plt.xlabel("T (min)")
        plt.axvline(x = 118, color = 'k', label = 'antibiotic')
        plt.show()

    return 0

def plotROIdata(dataframe, value_name,times,name="", *custom_values,batch = False, custom_keys = None,antibiotic = 0):
    names = dataframe['ROI name']
    unique_names = names.unique()
    branchtimes = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        branchtimes[i] = dataframe[dataframe['ROI name'] == i]["Times"].to_list()
    brancharray = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        brancharray[i] = dataframe[dataframe['ROI name'] == i][value_name].to_list()
    data = deepcopy(brancharray)
    if custom_values != ():
        data = custom_values[0]
    else:
        for i in branchtimes.keys():
            data[i] = [branchtimes[i],brancharray[i]]
    min_time = min(times)
    max_time = max(times)
    c_times = [(n-min_time)/(max_time-min_time) for n in times]
    colorbranchtimes = data.copy()
    for i in colorbranchtimes.keys():
        colorbranchtimes[i] = times.index(min(colorbranchtimes[i][0]))
    initial_keys = []
    for i in data.keys():
        if '.' not in i:
            initial_keys.append(i)
    if custom_keys is not None:
        initial_keys = custom_keys
    for j in initial_keys:
        for i in data.keys():
            if j in i:
                if j != i:
                    parent_index = i[:-1]
                    if parent_index[-1] == '.':
                        parent_index = parent_index[:-1]
                    if parent_index in data.keys():
                        parent_endtime = data[parent_index][0][-1]
                        parent_endvalue = data[parent_index][1][-1]
                        data[i][0] = [parent_endtime]+data[i][0]
                        data[i][1] = [parent_endvalue]+data[i][1]
                        
                col =  plt.cm.viridis(c_times[colorbranchtimes[i]])
                plt.plot(data[i][0],data[i][1],c =  col, ms = 1,alpha = 0.5)
                plt.axvline(x = data[i][0][0],color = col,label = 'Divisions', ls = '--')
        
                plt.xlabel("T (min)") 
                plt.title(name)
        plt.axvline(x = antibiotic , color = 'k', label = 'antibiotic')
        if not batch:
            plt.show()
            
    if  batch:
        plt.show()
        
    return 0

def getROIdata(dataframe,value_name,times):
    names = dataframe['ROI name']
    unique_names = names.unique()
    branchtimes = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        branchtimes[i] = dataframe[dataframe["ROI name"] == i]["Times"].to_list()
    brancharray = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        brancharray[i] = dataframe[dataframe["ROI name"] == i][value_name].to_list()
    for i in brancharray.keys():
        brancharray[i] = [branchtimes[i],brancharray[i]]
    return brancharray

def getTMdata(dataframe,value_name,times):
    names = dataframe['LABEL'].copy()
    unique_names = names.unique()
    branchtimes = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        branchtimes[i] = dataframe[dataframe["LABEL"] == i]["TIMES"].to_list()
    brancharray = {el:0 for el in unique_names}
    for i in branchtimes.keys():
        brancharray[i] = dataframe[dataframe["LABEL"] == i][value_name].to_list()
    for i in brancharray.keys():
        brancharray[i] = [branchtimes[i],brancharray[i]]
    return brancharray

def importexperiment(times,ACpath,name,dims,antibiotic=0,backpath = None,corrpath = None):
    ROIAC = pd.read_csv(ACpath)
    if backpath is not None:
        back = pd.read_csv(backpath)
        back = back['Median'].to_numpy().tolist()
        
    else: 
        back = [1 for x in times]
    ROIRes = ROIAC.copy(deep=True)
    interesting_values2 = ['Kurt','Skew','RawIntDen','Slice']
    ROIRes = processROIs(ROIRes,interesting_values2,times)
    cellist = getROIcells(ROIRes,times)
    exp = experiment(name,dims,times,cellist,antibiotic,backgrounds=back)
    exp.addsolidity()
    exp.getdensity()
    exp.normalizecellattr('color')
    if corrpath is not None:
        corr = pd.read_csv(corrpath)
        corr = processROIs(corr,interesting_values2,times)
    return exp
#%% ### IMPORT DATA ###





feb17_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_17-CST/ResultsACfilt-2.csv"
times_17feb = [0, 2, 3, 7, 8, 11, 15, 16, 17, 20, 23, 27, 31,  35, 39, 42, 51, 53, 61, 65, 66, 69, 72, 74, 76, 79, 80, 81, 85, 87, 89, 90, 93, 96, 97, 99, 102, 105, 107, 108, 109, 110, 113, 118, 126, 128, 131, 135]
feb17_dims = (372,376)
feb17_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_17-CST/ResultsACback.csv"
feb17_exp = importexperiment(times_17feb,feb17_path,'17 feb 20 mgL',feb17_dims,antibiotic = 33,backpath = feb17_bckg)

feb18one_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_18-CST1/ResultsACfilt-2.csv"
times_18febone = [0, 3, 4, 8, 11, 14, 17, 18, 21, 25, 29, 33, 37, 39, 43, 46, 47, 51, 53, 58, 61, 64, 66, 69, 73, 76, 77, 79, 80, 82, 86, 88, 90, 93, 94, 97, 100, 102, 104, 108, 117, 119, 123, 126, 160]
feb18one_dims = (372,376)
feb18one_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_18-CST1/ResultsACback.csv"
feb18one_exp = importexperiment(times_18febone,feb18one_path,'18 feb(1) ? mgL',feb18one_dims,antibiotic = 55,backpath = feb18one_bckg)

feb18two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_18-CST2/ResultsACfilt-2.csv"
times_18febtwo = [0, 3, 4, 9, 12, 15, 18, 19, 22, 25, 29, 32, 37, 41, 44, 45, 48, 51, 53, 54, 57, 58, 60, 62, 63, 67, 69, 72, 74, 76, 78, 81, 82, 84, 85, 88, 90, 98, 100, 102, 104, 110, 111, 112, 115]
feb18two_dims = (372,376)
feb18two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_18-CST2/ResultsACback.csv"
feb18two_exp = importexperiment(times_18febtwo,feb18two_path,'18 feb(2) ? mgL',feb18two_dims,antibiotic =55,backpath = feb18two_bckg)

feb19one_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST1/ResultsACfilt-2.csv"
times_19febone = [0, 7, 9, 10, 14, 17, 20, 22, 26, 29, 31, 33, 35, 37, 39, 46, 50, 51, 54, 59, 62, 68, 71, 73, 78, 84, 107]
feb19one_dims = (372,376)
feb19one_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST1/ResultsACback.csv"
feb19one_exp = importexperiment(times_19febone,feb19one_path,'19 feb(1) 10 mgL',feb19one_dims,antibiotic = 5,backpath = feb19one_bckg)

feb19two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST2/ResultsACfilt-2.csv"
times_19febtwo = [0, 4, 7, 10, 13, 14, 16, 19, 20, 22, 26, 29, 32, 33, 37, 38, 40, 45, 47, 50, 52, 57, 58, 61, 64, 66, 67, 70, 72, 75, 78, 79, 81, 83, 84, 86, 94, 95, 98, 100, 105, 107, 111, 112, 117, 118, 122, 126]
feb19two_dims = (372,376)
feb19two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST2/ResultsACback.csv"
feb19two_exp = importexperiment(times_19febtwo,feb19two_path,'19 feb(2) 10 mgL',feb19two_dims,antibiotic = 59,backpath = feb19two_bckg)

feb20two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsCorr.csv"
times_20febtwo = [0, 3, 7, 11, 14, 18, 22, 24, 33, 40, 42, 45, 49, 50, 51, 54, 58, 60, 63, 66, 67, 70, 73, 76, 79, 82, 83, 87, 88, 89]
feb20two_dims = (372,376)
feb20two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsCorrback.csv"
feb20two_exp = importexperiment(times_20febtwo,feb20two_path,'20 feb(2) 5 mgL',feb20two_dims,antibiotic = 34,backpath = feb20two_bckg)

jul12_path = "/Users/victorionescu/Desktop/imagini si rezultate/2024_07_12-CST/ResultsACfilt-2.csv"
times_12jul = [0, 6, 12, 17, 21, 26, 34, 40, 45, 61, 67, 75, 91, 97, 105, 113, 120, 136, 138, 151, 161]
jul12_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2024_07_12-CST/ResultsACback.csv"
jul12_dims = (323,496)
jul12_exp = importexperiment(times_12jul,jul12_path,'12 jul 10 mgL',jul12_dims,antibiotic  =26,backpath=jul12_bckg)


nov4_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_11_04-CTRL/ResultsACfilt-2.csv'
times_4nov = [0, 4, 16, 29, 33, 39, 41, 45, 51, 54, 61, 69, 76, 84]
nov4_dims = (352,342)
nov4_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_11_04-CTRL/ResultsACback.csv'
nov4_exp = importexperiment(times_4nov,nov4_path,'4 noi ctrl',nov4_dims,antibiotic = 0,backpath=nov4_bckg)


nov6_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_11_06-CTRL/ResultsACfilt-2.csv'
times_6nov = [0,9,16,23,26,30,37,43,45,61,70,74,77,81,88,96,99,103,111,118,123,129,133,141,149,154,161,168,174,181]
nov6_dims = (329,316)
nov6_bckg ='/Users/victorionescu/Desktop/imagini si rezultate/2024_11_06-CTRL/ResultsACback.csv'
nov6_exp = importexperiment(times_6nov,nov6_path,'6 noi ctrl',nov6_dims,antibiotic = 0,backpath=nov6_bckg)


nov12_path='/Users/victorionescu/Desktop/imagini si rezultate/2024_11_12-CST/ResultsACfilt-2.csv'
times_12nov = [0,1,5,6,7,13,16,19,25,26,31,36,38,39,42,45,47,51,55,56,65,66,67,68,70,72,78,79,81,83,88,93,96,98,100,103,106,107,112,114,118,121,122,125,131,136,143,152,162]
nov12_dims = (372,376)
nov12_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_11_12-CST/ResultsACback.csv'
nov12_exp = importexperiment(times_12nov,nov12_path,'12 noi 10 mgL',nov12_dims,antibiotic=32,backpath=nov12_bckg)

nov5_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_11_05-CST/ResultsACfilt-2.csv'
times_5nov = [0,2,10,14,17,23,29,34,38,42,47,52,56,62,69,75,79,85,88,93,95,104,114]
nov5_dims = (329,316)
nov5_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_11_05-CST/ResultsACback.csv'
nov5_exp = importexperiment(times_5nov,nov5_path,'5 noi 5 mgL',nov5_dims,antibiotic=20,backpath=nov12_bckg)

dec6_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_06-CTRL/ResultsACfilt-2.csv'
times_6dec = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93]
dec6_dims = (372,376)
dec6_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_06-CTRL/ResultsACback.csv'
dec6_exp = importexperiment(times_6dec,dec6_path,'6 dec ctrl',dec6_dims,antibiotic=0,backpath=dec6_bckg)


dec11one_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_11-CST1/ResultsACfilt-2.csv'
dec11one_dims = (372,376)
times_11decone = [0, 3, 6, 8, 9, 12, 16, 17, 19, 22, 24, 25, 26, 28, 30, 35, 37, 40, 42, 46, 49, 50]
dec11one_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_11-CST1/ResultsACback.csv'
dec11one_exp = importexperiment(times_11decone,dec11one_path,'11 dec (1) ?mgL',dec11one_dims,antibiotic=63,backpath=dec11one_bckg)

dec11two_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_11-CST2/ResultsACfilt-2.csv'
dec11two_dims = (372,376)
times_11dectwo = [0, 3, 4, 6, 9, 12, 13, 15, 17, 21, 22, 24, 27, 30, 33, 37, 39, 40, 42, 45, 48, 50, 51, 56, 59, 60, 69, 77, 79, 81, 83, 88]
dec11two_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_11-CST2/ResultsACback.csv'
dec11two_exp = importexperiment(times_11dectwo,dec11two_path,'11 dec (2) ?mgL',dec11two_dims,antibiotic=70,backpath=dec11two_bckg)

dec10_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_10-CST/ResultsACfilt-2.csv'
dec10_dims = (372,376)
times_10dec = [0, 3,6,9,12,15,16,18,21,24,27,30,33,36,39,42,52,59,62,65,68,71,74,77,80,83,86,89,92,95,98,101,104,107,112,115,118,125,128,131,134,137,140,143,146,149,152,155,158,161,164,167,170,173,176,179,182,184]
dec10_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_10-CST/ResultsACback.csv'
dec10_exp = importexperiment(times_10dec,dec10_path,'10 dec ?mgL',dec10_dims,antibiotic=63,backpath=dec10_bckg)

dec18_path ='/Users/victorionescu/Desktop/imagini si rezultate/2024_12_18-CST/ResultsACfilt-2.csv'
dec18_dims = (372,376)
times_18dec = [0, 3, 6, 10, 13, 16, 19, 22, 25, 28, 30, 36, 37, 49, 57, 60, 61, 62, 65, 66, 69, 72, 73, 75, 78, 81, 82, 84, 85, 88, 89, 90, 92, 93, 96, 101, 103, 104, 106, 109, 112, 115, 118, 119, 134]
dec18_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_18-CST/ResultsACback.csv'
dec18_exp = importexperiment(times_18dec,dec18_path,'18 dec ?mgL',dec18_dims,antibiotic=0,backpath=dec18_bckg) # concentration + time


ian16_path = '/Users/victorionescu/Desktop/imagini si rezultate/2025_01_16-CTRL/ResultsACfilt-2.csv'
ian16_dims = (372,376)
times_16ian = [0, 3, 7, 9, 12, 16, 19, 21, 25, 28, 32, 36, 44, 48, 54, 57, 60, 65, 67, 69, 72, 75, 76, 78, 81, 84, 86, 87, 92, 94, 98, 99, 101, 108, 111, 112, 116, 119, 122, 126, 128]
ian16_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2025_01_16-CTRL/ResultsACback.csv'
ian16_exp = importexperiment(times_16ian,ian16_path,'16 ian ctrl',ian16_dims,antibiotic=0,backpath=ian16_bckg)

ian17_path = '/Users/victorionescu/Desktop/imagini si rezultate/2025_01_17-CST/ResultsACfilt-2.csv'
ian17_dims = (372,376)
times_17ian = [0, 6, 9, 12, 19, 23, 25, 27, 28, 32, 34, 37, 43, 44, 51, 55, 56, 58, 59, 63, 66, 68, 70, 73, 75, 77, 79, 81, 84, 87, 92, 93, 96, 98, 101, 105, 108, 111, 114, 115, 117, 120, 121, 123, 124, 127, 130, 132, 141, 158]
ian17_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2025_01_17-CST/ResultsACback.csv'
ian17_exp = importexperiment(times_17ian,ian17_path,'17 ian 2.5 mgL',ian17_dims,antibiotic=49,backpath=ian17_bckg)

feb25one_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST1/ResultsACfilt-2.csv"
times_25febone = [0, 8, 14, 18, 22, 24, 30, 33, 35, 37, 45, 53, 56, 58, 61, 62, 66, 68, 71, 73, 74, 75, 78, 81, 84, 87, 90, 93, 94, 97, 100, 102, 105, 106, 110, 114, 117, 121, 124, 125, 133]
feb25one_dims = (372,376)
feb25one_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST1/ResultsACback.csv"
feb25one_exp = importexperiment(times_25febone,feb25one_path,'25 feb(1) 5 mgL',feb25one_dims,antibiotic = 64,backpath = feb25one_bckg)

feb25two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST2/ResultsAcfilt-2.csv"
times_25febtwo = [ 0, 2, 5, 7, 10, 22, 26, 28, 29, 33, 36, 40, 41, 44, 45, 49, 52, 55, 58, 62, 65, 68, 69, 72, 76, 81, 82, 84, 85, 88, 91, 97, 99, 100, 105, 107, 109, 111, 112, 115, 124]
feb25two_dims = (372,376)
feb25two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST2/ResultsACback.csv"
feb25two_exp = importexperiment(times_25febtwo,feb25two_path,'25 feb(2) 2.5 mgL',feb25two_dims,antibiotic = 67,backpath = feb25two_bckg)

mar3_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_03_03-CST/ResultsAcfilt-2.csv"
times_3mar = [ 0, 5, 9, 10, 13, 15, 19, 21, 22, 32, 38, 39, 44, 51, 52, 54, 55, 58, 61, 64, 65, 67, 71, 72, 74, 77, 80, 81, 85, 86, 88, 97, 98, 102, 103, 105, 106, 109, 110, 112, 115, 118, 121, 123, 126, 131, 133, 134]
mar3_dims = (372,376)
mar3_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_03_03-CST/ResultsACback.csv"
mar3_exp = importexperiment(times_3mar,mar3_path,'3 mar 1.25 mgL',mar3_dims,antibiotic = 67,backpath = mar3_bckg)

#%%
feb19one_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST1/ResultsACfilt-2.csv"
times_19febone = [0, 7, 9, 10, 14, 17, 20, 22, 26, 29, 31, 33, 35, 37, 39, 46, 50, 51, 54, 59, 62, 68, 71, 73, 78, 84, 107]
feb19one_dims = (372,376)
feb19one_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST1/ResultsACback.csv"
feb19one_exp = importexperiment(times_19febone,feb19one_path,'19 feb(1) 10 mgL',feb19one_dims,antibiotic = 5,backpath = feb19one_bckg)

feb19two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST2/ResultsACfilt-2.csv"
times_19febtwo = [0, 4, 7, 10, 13, 14, 16, 19, 20, 22, 26, 29, 32, 33, 37, 38, 40, 45, 47, 50, 52, 57, 58, 61, 64, 66, 67, 70, 72, 75, 78, 79, 81, 83, 84, 86, 94, 95, 98, 100, 105, 107, 111, 112, 117, 118, 122, 126]
feb19two_dims = (372,376)
feb19two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_19-CST2/ResultsACback.csv"
feb19two_exp = importexperiment(times_19febtwo,feb19two_path,'19 feb(2) 10 mgL',feb19two_dims,antibiotic = 59,backpath = feb19two_bckg)

feb20two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsAcfilt-2.csv"
times_20febtwo = [0, 3, 7, 11, 14, 18, 22, 24, 33, 40, 42, 45, 49, 50, 51, 54, 58, 60, 63, 66, 67, 70, 73, 76, 79, 82, 83, 87, 88, 89]
feb20two_dims = (372,376)
feb20two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsACback.csv"
feb20two_exp = importexperiment(times_20febtwo,feb20two_path,'20 feb(2) 5 mgL',feb20two_dims,antibiotic = 34,backpath = feb20two_bckg)

feb25one_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST1/ResultsACfilt-2.csv"
times_25febone = [0, 8, 14, 18, 22, 24, 30, 33, 35, 37, 45, 53, 56, 58, 61, 62, 66, 68, 71, 73, 74, 75, 78, 81, 84, 87, 90, 93, 94, 97, 100, 102, 105, 106, 110, 114, 117, 121, 124, 125, 133]
feb25one_dims = (372,376)
feb25one_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST1/ResultsACback.csv"
feb25one_exp = importexperiment(times_25febone,feb25one_path,'25 feb(1) 5 mgL',feb25one_dims,antibiotic = 64,backpath = feb25one_bckg)

feb25two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST2/ResultsAcfilt-2.csv"
times_25febtwo = [ 0, 2, 5, 7, 10, 22, 26, 28, 29, 33, 36, 40, 41, 44, 45, 49, 52, 55, 58, 62, 65, 68, 69, 72, 76, 81, 82, 84, 85, 88, 91, 97, 99, 100, 105, 107, 109, 111, 112, 115, 124]
feb25two_dims = (372,376)
feb25two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_25-CST2/ResultsACback.csv"
feb25two_exp = importexperiment(times_25febtwo,feb25two_path,'25 feb(2) 2.5 mgL',feb25two_dims,antibiotic = 67,backpath = feb25two_bckg)

mar3_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_03_03-CST/ResultsAcfilt-2.csv"
times_3mar = [ 0, 5, 9, 10, 13, 15, 19, 21, 22, 32, 38, 39, 44, 51, 52, 54, 55, 58, 61, 64, 65, 67, 71, 72, 74, 77, 80, 81, 85, 86, 88, 97, 98, 102, 103, 105, 106, 109, 110, 112, 115, 118, 121, 123, 126, 131, 133, 134]
mar3_dims = (372,376)
mar3_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_03_03-CST/ResultsACback.csv"
mar3_exp = importexperiment(times_3mar,mar3_path,'3 mar 1.25 mgL',mar3_dims,antibiotic = 67,backpath = mar3_bckg)

dec6_path = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_06-CTRL/ResultsACfilt-2.csv'
times_6dec = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93]
dec6_dims = (372,376)
dec6_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_12_06-CTRL/ResultsACback.csv'
dec6_exp = importexperiment(times_6dec,dec6_path,'6 dec ctrl',dec6_dims,antibiotic=0,backpath=dec6_bckg)

nov12_path='/Users/victorionescu/Desktop/imagini si rezultate/2024_11_12-CST/ResultsACfilt-2.csv'
times_12nov = [0,1,5,6,7,13,16,19,25,26,31,36,38,39,42,45,47,51,55,56,65,66,67,68,70,72,78,79,81,83,88,93,96,98,100,103,106,107,112,114,118,121,122,125,131,136,143,152,162]
nov12_dims = (372,376)
nov12_bckg = '/Users/victorionescu/Desktop/imagini si rezultate/2024_11_12-CST/ResultsACback.csv'
nov12_exp = importexperiment(times_12nov,nov12_path,'12 noi 10 mgL',nov12_dims,antibiotic=32,backpath=nov12_bckg)


#%%
mx = 0
for c in feb20two_exp.cells:
    
    if max([c.max[t] for t in c.times]) > mx:
        mx = max([c.max[t] for t in c.times])
print(mx)
#%%
explist = [nov5_exp,nov12_exp,dec10_exp,dec11one_exp,dec11two_exp,nov4_exp,nov6_exp,dec6_exp,ian16_exp,ian17_exp,feb17_exp,feb18one_exp,feb18two_exp,jul12_exp,feb19one_exp,feb19two_exp,feb20two_exp]

anticolors = mpl.colormaps['hot'].resampled(len(explist))
controlcolors = mpl.colormaps['viridis'].resampled(len(explist[:9]))
fig,ax = plt.subplots(2,figsize =(15,25))

for i,exp in enumerate(explist):
    maxgen = 0
    for c in exp.cells:
        if c.node.generation>maxgen:
            maxgen = c.node.generation
    if i<9:
        col = controlcolors(i)
    
        histsum = np.zeros(len(exp.cells[0].histogram[0]))
        for c in exp.cells:
                if exp.antibiotic < c.times[-1] and c.duration >15:
                    diffhist = [x-y for x,y in zip(c.histogram[c.times[-1]],c.histogram[c.times[0]])]
                    histsum += diffhist
                    
        histsum /=len(exp.cells)
        ax[0].plot(exp.cells[0].bins[0],histsum,c = col,label = exp.name,alpha = 0.8)
        print(f"Experiment:{exp.name}\nMax: {np.max(histsum)}\nMin:{np.min(histsum)}\nSpan:{np.max(histsum)-np.min(histsum)}\n----------\n")
    else:
        
        col = anticolors(i-9)
    
        histsum = np.zeros(len(exp.cells[0].histogram[0]))
        for c in exp.cells:
                if exp.antibiotic < c.times[-1] and c.duration >15:
                    diffhist = [x-y for x,y in zip(c.histogram[c.times[-1]],c.histogram[c.times[0]])]
                    histsum += diffhist
                    
        histsum /=len(exp.cells)
        ax[1].plot(exp.cells[0].bins[0],histsum,c = col,label = exp.name,alpha = 0.8)
        print(f"Experiment:{exp.name}\nMax: {np.max(histsum)}\nMin:{np.min(histsum)}\nSpan:{np.max(histsum)-np.min(histsum)}\n----------\n")
ax[0].legend(fancybox=True,loc='best',fontsize = 20)
ax[1].legend(fancybox=True,loc='best',fontsize = 20)
ax[0].set_title("Control/neclar")
ax[0].set_ylim(-0.015,0.025)
ax[1].set_ylim(-0.015,0.025)
ax[1].set_title("Antibiotic")
ax[1].set_xlabel('Intensitate AC',size = 15)
ax[1].set_ylabel('Diferenta intre histograme (final-start)',size = 15)
ax[0].set_xlabel('Intensitate AC',size = 15)
ax[0].set_ylabel('Diferenta intre histograme (final-start)',size = 15)

plt.show()
#%%
antilist = [jul12_exp,ian17_exp,feb17_exp,feb18one_exp,feb18two_exp,feb19one_exp,feb19two_exp,feb20two_exp]
anticolors = mpl.colormaps['hot'].resampled(len(explist))
plt.figure(figsize=(20,10))
custom_lines = []
names = []
for i,exp in enumerate(explist):
    if 'mgL' in exp.name:
        
        clist = [c for c in exp.cells if exp.times[-1] in c.times]
        col = anticolors(i-9)
    else:
        continue
        clist = exp.cells
        col = controlcolors(i)
    for c in clist:
        plt.plot(c.times,[c.ratio[t]-c.ratio[c.times[0]] for t in c.times],c = col)
    if exp.name == '11 dec (2) ?mgL':
        print(f'experiment name: {exp.name}')
        for c in clist:
            [print(c.mean[t]-c.mean[c.times[0]]) for t in c.times]
    plt.title(exp.name)
    custom_lines.append(mpl.lines.Line2D([0], [0], color=col, lw=4))
    names.append(exp.name)

plt.legend(custom_lines,names)
plt.show()
#%%
feb20two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsAcfilt-2.csv"
times_20febtwo = [0, 3, 7, 11, 14, 18, 22, 24, 33, 40, 42, 45, 49, 50, 51, 54, 58, 60, 63, 66, 67, 70, 73, 76, 79, 82, 83, 87, 88, 89]
feb20two_dims = (372,376)
feb20two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsACback.csv"
feb20two_exp = importexperiment(times_20febtwo,feb20two_path,'20 feb(2) 5 mgL',feb20two_dims,antibiotic = 34,backpath = feb20two_bckg)

feb20two_path = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsCorr.csv"
times_20febtwo = [0, 3, 7, 11, 14, 18, 22, 24, 33, 40, 42, 45, 49, 50, 51, 54, 58, 60, 63, 66, 67, 70, 73, 76, 79, 82, 83, 87, 88, 89]
feb20two_dims = (372,376)
feb20two_bckg = "/Users/victorionescu/Desktop/imagini si rezultate/2025_02_20-CST2/ResultsCorrback.csv"
feb20two_exp_corr = importexperiment(times_20febtwo,feb20two_path,'20 feb(2) 5 mgL',feb20two_dims,antibiotic = 34,backpath = feb20two_bckg)
#%%
exp = feb20two_exp
cellist_AC = [c for c in exp.cells if c.times[-1] >= exp.antibiotic]
colors = mpl.colormaps['rainbow'].resampled(len(cellist_AC))
for i,c in enumerate(cellist_AC):
    
    plt.plot(c.times,[(c.interior.mean[t]-exp.backgrounds[t])/(c.interior.mean[t]+exp.backgrounds[t]) for t in c.times],label = c.name,c = colors(i))
plt.legend(fancybox = True,loc='best',prop={'size':7})

plt.ylabel("Contrast interior AC/fundal AC")
plt.title("Imagini AC")
plt.xlabel("T(min)")
plt.show()
exp = feb20two_exp_corr
cellist_corr = [c for c in exp.cells if c.times[-1] >= exp.antibiotic]
colors = mpl.colormaps['rainbow'].resampled(len(cellist_corr))
for i,c in enumerate(cellist_corr):
    
    plt.plot(c.times,[(c.interior.mean[t]-exp.backgrounds[t])/(c.interior.mean[t]+exp.backgrounds[t]) for t in c.times],label = c.name,c = colors(i))
plt.legend(fancybox = True,loc='best',prop={'size':7})

plt.xlabel("T(min)")
plt.title("Imagini Corelatie")
plt.ylabel("Contrast interior corelatie/fundal corelatie")
plt.show()


#%%
for c_AC,c_corr in zip(cellist_AC,cellist_corr):
    fig,ax = plt.subplots(2,figsize = (10,10))
    colors = mpl.colormaps['rainbow'].resampled(len(c_AC.times))
    for i in range(0,len(c_AC.times),2):
        t = c_AC.times[i]
        ax[0].plot(c_AC.bins[t],c_AC.histogram[t],c = colors(i))
        ax[1].plot(c_corr.bins[t],c_corr.histogram[t],c = colors(i))
        print(i)
    fig.suptitle(c_AC.name)
    ax[0].set_title('Amplitudine (AC)')
    ax[1].set_title('Corelatie')
    ax[0].set_xlim(0,32000)
    ax[1].set_xlim(0,18000)

    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,c_AC.times[-1]-c_AC.times[0]), cmap='rainbow'), ax=ax,label = "T (min)",)
    plt.show()        
#%%
print(len(feb20two_exp.cells))
explist = [feb19one_exp,feb19two_exp,feb20two_exp,feb25one_exp,feb25two_exp,mar3_exp,dec6_exp,nov12_exp]
fig,ax = plt.subplots(3,figsize = (10,15))
for exp in explist:
    ax[0].plot(exp.times,[exp.backgrounds[t] for t in exp.times],label = exp.name+"-back")
    avgint = []
    stdint = []
    for t in exp.times:
        avg = []
        for c in exp.cells:
            if t in c.times:
                avg.append((c.interior.mean[t]-exp.backgrounds[t])/(c.interior.mean[t]+exp.backgrounds[t]))
        avgint.append(np.mean(avg))
        stdint.append(np.std(avg))
    ax[1].plot(exp.times,avgint,label = exp.name+"-cellint")
    
    ax[2].plot([t-exp.antibiotic for t in exp.times],avgint,label = exp.name+"-contrast")
    ax[2].fill_between([t-exp.antibiotic for t in exp.times],[x-y for x,y in zip(avgint,stdint)],[x+y for x,y in zip(avgint,stdint)],alpha = 0.1)
    ax[2].vlines(0,0.1,0.5)
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[2].set_ylim(0.15,0.45)
plt.show()
#%%
# goodlist = [jul12_exp,feb20two_exp,ian16_exp,feb19one_exp,feb19two_exp,feb17_exp,feb18one_exp,dec6_exp]
# goodlist = [feb20two_exp,ian16_exp,feb19one_exp,feb19two_exp,dec6_exp,feb17_exp]
goodlist = [feb20two_exp,feb25two_exp,mar3_exp,feb25one_exp]
for exp in goodlist:
    meanback = np.mean([exp.backgrounds[t] for t in exp.times])
    stdback = np.std([exp.backgrounds[t] for t in exp.times])
    print(meanback)
    print(stdback)
for exp in goodlist:
    counter = 0
    for c in exp.cells:
        if c.times[-1]>= exp.antibiotic and counter <10 and c.duration >10:
            cols = mpl.colormaps['viridis'].resampled(len(c.times))
            for i,t in enumerate(c.times):
               
                plt.plot(c.bins[t],c.interior.histogram[t],c = cols(i))
                
            plt.title(f'Exp: {exp.name}; Cell: {c.name}')
            maxelarg = np.nonzero(c.interior.histogram[t])[0][-1]
            maxelement = c.bins[t][maxelarg]
            plt.xlim(c.mean[t]-c.std[t],c.mean[t]+c.std[t])
            plt.show()
            counter+=1
colors = mpl.colormaps['rainbow'].resampled(len(goodlist))
for i,exp in enumerate(goodlist):
    counter = 0
    for c in exp.cells:
        if c.times[-1]>= exp.antibiotic and c.duration >0:
            value = [np.sum([c.histogram[t][i]*500  for i in range(len(c.bins[t])) if c.bins[t][i] >c.hmean[t]-c.hstd[t] and c.bins[t][i] <c.mean[t]+c.std[t]]) for t in c.times]
            plt.plot(c.times[:],value,c = colors(i))
            
            counter+=1
    plt.title(exp.name+"; Anti T: "+str(exp.antibiotic))
plt.legend([mpl.lines.Line2D([0], [0], color=colors(i), lw=4) for i in range(len(goodlist))],[e.name for e in goodlist])
plt.show()
#%%
cel = [c for c in feb20two_exp.cells if c.name == 'Track_3.aa'][0]
times = [40,50,feb20two_exp.times[-1]]
for t in times:
    plt.plot(cell.bins[t],cell.histogram[t])
    plt.xlim(0,20000)
    plt.title(cell.name+"; Time = "+str(t))
    plt.ylim(0,0.09)
    plt.show()
for t in times:
    plt.plot(cell.bins[t],cell.histogram[t],label = "T ="+str(t))
    plt.xlim(0,20000)
    plt.title(cell.name)
plt.legend()
plt.show()
#%%
# goodlist = [jul12_exp,feb20two_exp,feb19one_exp,feb19two_exp,ian16_exp,feb18one_exp,feb18two_exp]
goodlist = [feb25two_exp]
# ctrllist = [dec6_exp,nov6_exp,nov4_exp]
fig,ax = plt.subplots(2,figsize = (10,10),sharey = True)
colors = mpl.colormaps['rainbow'].resampled(len(goodlist))
for i,exp in enumerate(goodlist):
    meandictanti = {t:[] for t in exp.times}
    meandictcontrol = {t:[] for t in exp.times}
    valuename = "Contrast interior fundal"
    for c in exp.cells:
        value = [ np.sum([c.histogram[t][i] if c.bins[t][i] > c.mean[t]-c.std[t] and  c.bins[t][i] < c.mean[t]+c.std[t] for i in range(len(c.bins[t])]) for t in c.times]
        time = c.times
        
        if c.times[-1]>= exp.antibiotic and c.duration>10 and exp.antibiotic != 0:
            ax[0].plot(time,value,c = colors(i),alpha = 0.3)
            [meandictanti[t].append(v) for t,v in zip(time,value)]
        else:
            if c.duration >10:
                ax[1].plot(time,value,c=colors(i),alpha = 0.3)
                [meandictcontrol[t].append(v) for t,v in zip(time,value)]
    antitimes = [t for t in exp.times if meandictanti[t] != []]
    controltimes = [t for t in exp.times if meandictcontrol[t] != []]
    
    ax[0].errorbar(antitimes,[np.mean(meandictanti[t]) for t in antitimes],[np.std(meandictanti[t]) for t in antitimes],markersize = 20,c = colors(i),label = exp.name+'-Anti trendline')
    ax[1].errorbar(controltimes,[np.mean(meandictcontrol[t]) for t in controltimes],[np.std(meandictcontrol[t]) for t in controltimes],markersize = 20,c = colors(i),label = exp.name+'-Control trendline')
    ax[0].set_title("Anti: T="+str(exp.antibiotic))
    ax[1].set_title("Ctrl")
    ax[0].set_ylabel(valuename)
    ax[1].set_ylabel(valuename)
    ax[0].set_xlabel("T(min)")
    ax[1].set_xlabel("T(min)")
ax[1].legend()
ax[0].legend()
    
plt.show()
   
# for c in feb20two_exp.cells:
#     if c.times[-1]>= exp.antibiotic and c.duration>10:
#         plt.plot([t for t in c.times if c.maxpoledistance[t] !=0],[c.maxpoledistance[t]/c.major[t] for t in c.times if c.maxpoledistance[t]!=0])
        
# plt.show()

#%%
### Check duplicates
wrongcells = []
for exp in explist:
    for c in exp.cells:
        if np.unique(c.times).tolist() != c.times:
            print(exp.name)
            print(c.name)
            _,counts = np.unique(c.times,return_counts=True)
            index = np.argwhere(counts!=1)
            nonutimes = np.asarray(c.times)[index]
            #gets nonunique times
            #finds noutimes in exp.times
            
            timelist= []
            for el in nonutimes:
                timelist.append(el)
            imageno = [exp.times.index(e)+1 for e in timelist]
            print(imageno)
            wrongcells.append(c)
            
    
#%%
#explist = [feb19two_exp,feb19one_exp,feb17_exp,feb18one_exp]



for i,exp in enumerate(explist):
    maxgen=0
    for c in exp.cells:
        if c.node.generation>maxgen:
            maxgen = c.node.generation
    if i<8:
        col = controlcolors(i)
      
    else:
        col = anticolors(i-8)
    for c in exp.cells:
        if exp.antibiotic < c.times[-1] and c.duration >15 :
            plt.plot([t-c.times[0] for t in c.times],[c.mean[t] for t in c.times],c = col)
    plt.title(exp.name)         
    plt.show()
        

#%%
explist = [nov5_exp,nov12_exp,dec10_exp,dec11one_exp,dec11two_exp,nov4_exp,nov6_exp,dec6_exp,ian16_exp,ian17_exp,feb17_exp,feb18one_exp,feb18two_exp,jul12_exp,feb19one_exp,feb19two_exp,feb20two_exp]
colors = mpl.colormaps['rainbow'].resampled(len(explist))
plt.figure(figsize = (20,10))
for i,exp in enumerate(explist):
    for c in exp.cells:
        value = [c.interior.mean[t]/exp.backgrounds[t] for t in c.times]
        time = c.times
        if c.times[-1]>exp.antibiotic and c.duration >10:
            plt.plot(time,value,c=colors(i),label = exp.name,alpha = 0.8)
plt.legend([mpl.lines.Line2D([0], [0], color=colors(i), lw=4) for i in range(len(explist))],[e.name for e in explist])
plt.show()

#%%
for c in jul12_exp.cells:
    plt.plot(c.times,[c.ACsolidity[t] for t in c.times],c='b')
    
for c in feb17_exp.cells:
    plt.plot(c.times,[c.ACsolidity[t] for t in c.times],c='cyan')


for c in ian16_exp.cells:
    plt.plot(c.times,[c.ACsolidity[t]  for t in c.times],c='g')  
    
for c in feb18one_exp.cells:
    plt.plot(c.times,[c.ACsolidity[t]  for t in c.times],c='r')  
    
for c in feb18two_exp.cells:
    plt.plot(c.times,[c.ACsolidity[t]  for t in c.times],c='purple')  
    
custom_lines = [mpl.lines.Line2D([0], [0], color='b', lw=4),
                mpl.lines.Line2D([0], [0], color='r', lw=4)
                ]
plt.legend(custom_lines,["12 iulie 10mgL ","17 feb 20 mgL"])
plt.xlabel("T(min)")
plt.ylabel("Medie interior AC")
plt.title("Progresii per celula")

#%%
explist = [nov4_exp,nov5_exp,nov6_exp,nov12_exp,dec6_exp,dec10_exp,ian16_exp,ian17_exp]
dens = []
durations = []

for exp in explist:
    roots = exp.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
    
    for c in exp.cells:
        if c.node.generation >0 and c.node.generation < mx and c.duration >15:
            densities = np.mean([c.density[t] for t in c.times])
            slope = c.majorslope
            duration = c.duration
            if 'mgL'  in exp.name:
               color = 'r'
               if c.times[0]<exp.antibiotic or exp.name == '17 ian 1.5 mgL':
                   continue
            else:
                color = 'b'
                
            plt.scatter(densities,duration,c = color,alpha = 0.5)
            dens.append(densities)
            durations.append(duration)
plt.show()    
for exp in explist:
    exp.averagevalue('density')
    plt.scatter([exp.cellcount[t] for t in exp.times],exp.density_avg,label = exp.name)

plt.legend()
plt.show()
#%%
def findexp(c):
    exnames = ["nov6exp","nov12exp","dec6exp"]
    for i,ex in enumerate([nov6_exp,nov12_exp,dec6_exp]):
        if c in ex.cells:
            return exnames[i]
def flattenlist(l):
    return [x for xs in l for x in xs]


from sklearn.decomposition import PCA
from sklearn.cluster import dbscan
explist = [nov4_exp,nov5_exp,nov6_exp,nov12_exp,dec6_exp,dec10_exp,ian16_exp,ian17_exp]

nov12cs = [c for c in nov12_exp.cells if  c.node.generation >=0 and c.node.generation <7 and c.times[-1]-c.times[0] >0]
nov6cs = [c for c in nov6_exp.cells if  c.duration <80 and c.node.generation >0 and c.node.generation <7 and c.times[-1]-c.times[0] >0]
dec6cs = [c for c in dec6_exp.cells if  c.duration <80 and c.duration > 0 and c.node.generation >=0 and c.node.generation <7 and c.times[-1]-c.times[0] >0]
parameterarray  = np.zeros((len(nov6cs)+len(nov12cs)+len(dec6cs),7))
counter = 0
nov12ind = {}
nov6ind = {}
dec6ind = {}

for c in nov12cs:
    parameterarray[counter,0] = np.nan_to_num((c.major[c.times[-1]]-c.major[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,1] = np.nan_to_num((c.area[c.times[-1]]-c.area[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,2] = c.duration
    parameterarray[counter,3] = np.mean([c.ACsolidity[t] for t in c.times])
    parameterarray[counter,4] = np.mean([c.interior.mean[t]/c.contour.mean[t] for t in c.times])
    parameterarray[counter,5] = np.nan_to_num(np.mean(flattenlist([c.polesprom[t] for t in c.times])))
    parameterarray[counter,6] = np.mean([c.density[t] for t in c.times])
    nov12ind.update({counter:c})
    counter+=1
for c in nov6cs:
    parameterarray[counter,0] = np.nan_to_num((c.major[c.times[-1]]-c.major[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,1] = np.nan_to_num((c.area[c.times[-1]]-c.area[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,2] = c.duration
    parameterarray[counter,3] = np.mean([c.ACsolidity[t] for t in c.times])
    parameterarray[counter,4] = np.mean([c.interior.mean[t]/c.contour.mean[t] for t in c.times])
    parameterarray[counter,5] = np.mean(flattenlist([c.polesprom[t] for t in c.times]))
    parameterarray[counter,6] = np.mean([c.density[t] for t in c.times])
    nov6ind.update({counter:c})
    counter+=1
for c in dec6cs:
    parameterarray[counter,0] = np.nan_to_num((c.major[c.times[-1]]-c.major[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,1] = np.nan_to_num((c.area[c.times[-1]]-c.area[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,2] = c.duration
    parameterarray[counter,3] = np.mean([c.ACsolidity[t] for t in c.times])
    parameterarray[counter,4] = np.mean([c.interior.mean[t]/c.contour.mean[t] for t in c.times])
    parameterarray[counter,5] = np.mean(flattenlist([c.polesprom[t] for t in c.times]))
    parameterarray[counter,6] = np.mean([c.density[t] for t in c.times])
    dec6ind.update({counter:c})
    counter+=1
    
cellindexes = {}
cellindexes.update(nov12ind)
cellindexes.update(nov6ind)
cellindexes.update(dec6ind)



#%%

for i in range(6):
    parameterarray[:,i] -= np.mean(parameterarray[:,i])
    parameterarray[:,i] /= np.std(parameterarray[:,i])
    parameterarray = np.nan_to_num(parameterarray)
    
cores,labels =dbscan(parameterarray,eps = 0.9)
def findexp(c):
    exnames = ["nov6exp","nov12exp","dec6exp"]
    for i,ex in enumerate([nov6_exp,nov12_exp,dec6_exp]):
        if c in ex.cells:
            return exnames[i]
uniqlabels = np.unique(labels)
cellclusters = {}
for el in uniqlabels:
    print(f"Cluster: {el}")
    cellclusters.update({el:[]})
    for i,lb in enumerate(labels):
        
        if lb ==el:
            print(f"Cellname: {cellindexes[i].name}")
            print(f"Cellgen: {cellindexes[i].node.generation}")
            print(f"Cellexp: {findexp(cellindexes[i])}\n")
            cellclusters[el].append((cellindexes[i].name,cellindexes[i].node.generation,findexp(cellindexes[i])))
            
    print("--------------------\n")

pca = PCA(n_components=3)
cmap12 = [0 if c.node.generation <3 else 1 for c in nov12cs ]
cmap6 = [2 for c in nov6cs]
cmapdec6 = [3 for c in dec6cs]
cmap = mpl.colormaps['viridis'].resampled(4)

colors= cmap12+cmap6+cmapdec6
colors = cmap(colors)

pcares = pca.fit_transform(parameterarray)
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')

ax.scatter(pcares[:,0],pcares[:,1],pcares[:,2],c = colors,s = 30)
ax.view_init(elev=60, azim=100)
ax.set_zlim(-3,3)
ax.set_ylim(-5,6)
ax.set_xlim(-5,8)
plt.show()
#%%
for i,exp in enumerate(explist):
    plt.plot(exp.times,[exp.backgrounds[t] for t in exp.times],label = exp.name)
        

plt.legend()
plt.show()
for i,exp in enumerate(explist):
    for c in exp.cells:
        if i <5:
            color = 'g'
        else:
            color = 'r'
        plt.scatter(c.times,[c.interior.mean[t]/c.contour.mean[t] for t in c.times],c = color)
#%%
def flattenlist(l):
    return [x for xs in l for x in xs]

def findfullgens(exp):
    roots = exp.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
            
    return [1,mx-1]

def valuesextractor(exp,genlimits):
    valuedict = {"major slope":[],"area slope":[],"duration":[],"ACcontrast":[],"ACsolidity":[],"poles prominence":[],"poles distance slope":[]}
    
    
    for c in exp.cells:
        
        if not c.outlier and c.node.generation >= genlimits[0] and c.node.generation <= genlimits[1]:
        
            x = c.duration
            valuedict["duration"].append(x)
            x = np.mean([c.interior.mean[t]/c.contour.mean[t] for t in c.times])
            valuedict["ACcontrast"].append(x)
            x = np.mean([c.ACsolidity[t] for t in c.times])
            valuedict["ACsolidity"].append(x)
            x = np.mean(flattenlist([c.polesprom[t] for t in c.times]))
            valuedict["poles prominence"].append(x)
            if c.times[-1]-c.times[0] == 0:
                continue
            x = (c.area[c.times[-1]]-c.area[c.times[0]])/(c.times[-1]-c.times[0])
            valuedict["area slope"].append(x)
            x = (c.major[c.times[-1]]-c.major[c.times[0]])/(c.times[-1]-c.times[0])
            valuedict["major slope"].append(x)
            x = (c.maxpoledistance[c.times[-1]]-c.maxpoledistance[c.times[0]])/(c.times[-1]-c.times[0])
            valuedict["poles distance slope"].append(x)
        
    return valuedict

nov6_vals = valuesextractor(nov6_exp,findfullgens(nov6_exp))
dec6_vals = valuesextractor(dec6_exp,findfullgens(dec6_exp))
ian16_vals = valuesextractor(ian16_exp,findfullgens(ian16_exp))
nov4_vals = valuesextractor(nov4_exp,findfullgens(nov4_exp))

def boxplotvalues(expvalues,names):
    for k in expvalues[0].keys():
        vallists = [np.nan_to_num(it[k]) for it in expvalues]
        
        plt.violinplot(vallists)
        plt.ylabel(k)
        plt.show()
        

names = ["nov6","dec6","ian16","nov4"]
expvalues = [nov6_vals,dec6_vals,ian16_vals,nov4_vals]
control = {"major slope":[],"area slope":[],"duration":[],"ACcontrast":[],"ACsolidity":[],"poles prominence":[],"poles distance slope":[]}
for ex in expvalues:
    control["duration"] += ex["duration"]
    if ex is not ian16_vals:
        control["ACcontrast"] += ex["ACcontrast"]
        control["ACsolidity"] += ex["ACsolidity"]
        control["poles prominence"] += ex["poles prominence"]
    control["area slope"] += ex["area slope"]
    control["major slope"] += ex["major slope"]
    control["poles distance slope"] += ex["poles distance slope"]
    
nov5_vals = valuesextractor(nov5_exp,findfullgens(nov5_exp))
nov12_vals = valuesextractor(nov12_exp,findfullgens(nov12_exp))
dec10_vals = valuesextractor(dec10_exp,findfullgens(dec10_exp))
ian17_vals = valuesextractor(ian17_exp,findfullgens(ian17_exp))
expvalues = [nov5_vals,dec10_vals,ian17_vals,nov12_vals,control]
names = ["nov5","dec10","ian17","nov12","control"]
boxplotvalues(expvalues,names)
            
    

#%%
def plotsoftmax(exp,color = 'b'):
    for c in exp.cells:
        if c.node.generation >0 and c.node.generation<4:
            mxthresholds = [0.9*c.max[t] for t in c.times]
            mxhistelements = [[(el,b) for el,b in zip(c.histogram[t],c.bins[t]) if b >=th] for t,th in zip(c.times,mxthresholds)]
            mxvalues = [np.sum([el[0]*el[1] for el in tlist])/np.sum([el[0] for el in tlist]) for tlist in mxhistelements]
        
            plt.plot([t-c.times[0] for t in c.times],mxvalues,c = color)
        

       
plotsoftmax(dec6_exp)
plotsoftmax(nov6_exp,color = 'r')
plotsoftmax(nov12_exp,color = "purple")
plt.show()
    

#%%
means = []
tshift = 0
counter = 0
limit = len([c for c in dec6_exp.cells if c.node.generation ==3])
viridis = mpl.colormaps['rainbow'].resampled(limit)
for c in dec6_exp.cells:
    if  not c.outlier and c.duration < 70 and c.node.generation <7 and c.node.generation >0:
        values = [c.interior.mean[t]/c.contour.mean[t] for t in c.times]
        
        times = [t-tshift*c.times[0] for t in c.times]
        plt.plot(times,values,c = 'r',alpha = 0.5)
        plt.scatter(times,values,c = 'r',alpha = 0.5)
        #plt.scatter([times[0],times[-1]],[values[0],values[-1]],c='r')
        means += [np.median(c.polesprom[t]) for t in c.times]
        counter+=1

print(np.mean(means),np.std(means))
plt.xlabel("T(min)")
plt.ylabel("dPoli/major(DC)")
plt.title("6 dec control")


# plt.plot(dec6_exp.times,[dec6_exp.backgrounds[t] for t in dec6_exp.times])
plt.show()
means = []

for c in nov6_exp.cells:
    if not c.outlier and c.duration <70 and c.node.generation <7 and c.node.generation >0:
        values = [c.interior.mean[t]/c.contour.mean[t] for t in c.times]
        
        times = [t-tshift*c.times[0] for t in c.times ]
        plt.plot(times,values,c = 'b',alpha = 0.5)
        plt.scatter(times,values,c = 'b',alpha = 0.5)
        #plt.scatter([times[0],times[-1]],[values[0],values[-1]],c='b')
        means += [np.median(c.polesprom[t]) for t in c.times] 
        


# print(np.mean(means),np.std(means))
# plt.xlabel("T(min)")
# plt.ylabel("dPoli/major(DC)")
# plt.title("6 noi control")
plt.show()
# plt.plot(nov6_exp.times,[nov6_exp.backgrounds[t]for t in nov6_exp.times],c ='g')
# plt.plot(dec6_exp.times,[dec6_exp.backgrounds[t]for t in dec6_exp.times],c='k')
# plt.show()
# means = []

for c in nov12_exp.cells:
    if not c.outlier and c.node.generation <7 and c.node.generation >0:
        values = [c.interior.mean[t]/c.contour.mean[t] for t in c.times]
        times = [t-tshift*c.times[0] for t in c.times ]
        plt.plot(times,values,alpha = 0.5,c='purple')
        plt.scatter(times,values,alpha = 0.5,c='purple')
        #plt.scatter([times[0],times[-1]],[values[0],values[-1]],c='b')
        means += [c.interior.mean[t] for t in c.times]  

       

plt.xlabel("T(min)")
plt.ylabel("Distanta intre polii extremi")
plt.title("12 noi antibiotic")
plt.show()
#%%

def getsoftmax(c):
    mxthresholds = [0.9*c.interior.max[t] for t in c.times]
    mxhistelements = [[(el,b) for el,b in zip(c.interior.histogram[t],c.interior.bins[t]) if b >=th] for t,th in zip(c.times,mxthresholds)]
    mxvalues = [np.sum([el[0]*el[1] for el in tlist])/np.sum([el[0] for el in tlist]) for tlist in mxhistelements]
    mxvalues = [mx/c.contour.mean[t] for t,mx in zip(c.times,mxvalues)]
                                                     
    return mxvalues

def getsoftmin(c):
    mxthresholds = [1.5*c.interior.min[t] for t in c.times]
    mxhistelements = [[(el,b) for el,b in zip(c.interior.histogram[t],c.interior.bins[t]) if b <=th] for t,th in zip(c.times,mxthresholds)]
    mxvalues = [np.sum([el[0]*el[1] for el in tlist])/np.sum([el[0] for el in tlist]) for tlist in mxhistelements]
    mxvalues = [mx/c.contour.mean[t] for t,mx in zip(c.times,mxvalues)]
    return mxvalues

import csv
def flattenlist(l):
    return [x for xs in l for x in xs]
def makeangle(i):
    return np.arccos(i/10000-1)


valuearray = []
for c in nov6_exp.cells:
    if not c.outlier and c.duration < 80 and c.node.generation >0 and c.node.generation <4:
        
        value = np.mean([c.mean[t] for t in c.times])
        if value==0:
            continue
        
        valuearray.append(value)
        
        
valuearray2 = []
for c in dec6_exp.cells:
    if not c.outlier and c.duration < 80 and c.duration > 0 and c.node.generation >=0 and c.node.generation <3:
        
        value = np.mean([c.mean[t] for t in c.times])
        
        if value==0:
            continue
        
        valuearray2.append(value)
valuearray3 = []
for c in nov12_exp.cells:
    if not c.outlier  and c.node.generation >=0 and c.node.generation <3:
        
        value = np.mean([c.mean[t] for t in c.times])
        
        if value==0:
            continue
        
        valuearray3.append(value)
mean = np.mean(valuearray+valuearray2)
plt.boxplot([valuearray,valuearray2,valuearray3,valuearray+valuearray2],labels=["nov6exp",'dec6exp','nov12_exp','total'])
plt.ylabel('Soft maximum (>90%) din interiorul corr')
plt.show()
std = np.std(valuearray+valuearray2)
median = np.median(valuearray+valuearray2)
q1 = np.percentile(valuearray+valuearray2, 25)
q3 = np.percentile(valuearray+valuearray2, 75)
iqr = q3 - q1
valuename = "AR growth"
print(f"Value name: {valuename}")
print(f"Npoints: {len(valuearray+valuearray2)}")
print(f"Mean: {mean}")
print(f"STD: {std}")
print(f"Median: {median}")
print(f"IQR: {iqr}")
print(np.min(valuearray+valuearray2))
#%%
from sklearn.decomposition import PCA
from sklearn.cluster import dbscan

nov12cs = [c for c in nov12_exp.cells if  c.node.generation >=0 and c.node.generation <7 and c.times[-1]-c.times[0] >0]
nov6cs = [c for c in nov6_exp.cells if  c.duration <80 and c.node.generation >0 and c.node.generation <7 and c.times[-1]-c.times[0] >0]
dec6cs = [c for c in dec6_exp.cells if  c.duration <80 and c.duration > 0 and c.node.generation >=0 and c.node.generation <7 and c.times[-1]-c.times[0] >0]
parameterarray  = np.zeros((len(nov6cs)+len(nov12cs)+len(dec6cs),7))
counter = 0
nov12ind = {}
nov6ind = {}
dec6ind = {}

for c in nov12cs:
    parameterarray[counter,0] = np.nan_to_num((c.major[c.times[-1]]-c.major[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,1] = np.nan_to_num((c.area[c.times[-1]]-c.area[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,2] = c.duration
    parameterarray[counter,3] = np.mean([c.ACsolidity[t] for t in c.times])
    parameterarray[counter,4] = np.mean([c.interior.mean[t]/c.contour.mean[t] for t in c.times])
    parameterarray[counter,5] = np.nan_to_num(np.mean(flattenlist([c.polesprom[t] for t in c.times])))
    parameterarray[counter,6] = np.mean(c.density[t] for t in c.times)
    nov12ind.update({counter:c})
    counter+=1
for c in nov6cs:
    parameterarray[counter,0] = np.nan_to_num((c.major[c.times[-1]]-c.major[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,1] = np.nan_to_num((c.area[c.times[-1]]-c.area[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,2] = c.duration
    parameterarray[counter,3] = np.mean([c.ACsolidity[t] for t in c.times])
    parameterarray[counter,4] = np.mean([c.interior.mean[t]/c.contour.mean[t] for t in c.times])
    parameterarray[counter,5] = np.mean(flattenlist([c.polesprom[t] for t in c.times]))
    nov6ind.update({counter:c})
    counter+=1
for c in dec6cs:
    parameterarray[counter,0] = np.nan_to_num((c.major[c.times[-1]]-c.major[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,1] = np.nan_to_num((c.area[c.times[-1]]-c.area[c.times[0]])/(c.times[-1]-c.times[0]))
    parameterarray[counter,2] = c.duration
    parameterarray[counter,3] = np.mean([c.ACsolidity[t] for t in c.times])
    parameterarray[counter,4] = np.mean([c.interior.mean[t]/c.contour.mean[t] for t in c.times])
    parameterarray[counter,5] = np.mean(flattenlist([c.polesprom[t] for t in c.times]))
    dec6ind.update({counter:c})
    counter+=1
    
cellindexes = {}
cellindexes.update(nov12ind)
cellindexes.update(nov6ind)
cellindexes.update(dec6ind)



#%%

for i in range(6):
    parameterarray[:,i] -= np.mean(parameterarray[:,i])
    parameterarray[:,i] /= np.std(parameterarray[:,i])
    parameterarray = np.nan_to_num(parameterarray)
    
cores,labels =dbscan(parameterarray,eps = 0.9)
def findexp(c):
    exnames = ["nov6exp","nov12exp","dec6exp"]
    for i,ex in enumerate([nov6_exp,nov12_exp,dec6_exp]):
        if c in ex.cells:
            return exnames[i]
uniqlabels = np.unique(labels)
cellclusters = {}
for el in uniqlabels:
    print(f"Cluster: {el}")
    cellclusters.update({el:[]})
    for i,lb in enumerate(labels):
        if lb ==el:
            print(f"Cellname: {cellindexes[i].name}")
            print(f"Cellgen: {cellindexes[i].node.generation}")
            print(f"Cellexp: {findexp(cellindexes[i])}\n")
            cellclusters[el].append((cellindexes[i].name,cellindexes[i].node.generation,findexp(cellindexes[i])))
            
    print("--------------------\n")

pca = PCA(n_components=3)
cmap12 = [0 if c.node.generation <3 else 1 for c in nov12cs ]
cmap6 = [2 for c in nov6cs]
cmapdec6 = [3 for c in dec6cs]
cmap = mpl.colormaps['viridis'].resampled(4)

colors= cmap12+cmap6+cmapdec6
colors = cmap(colors)

pcares = pca.fit_transform(parameterarray)
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')

ax.scatter(pcares[:,0],pcares[:,1],pcares[:,2],c = colors,s = 30)
ax.view_init(elev=60, azim=100)
ax.set_zlim(-3,3)
ax.set_ylim(-5,6)
ax.set_xlim(-5,8)
plt.show()
    

#%%
pca = PCA(n_components=3)
cmap12 = [0 if c.node.generation <3 else 1 for c in nov12cs ]
cmap6 = [2 for c in nov6cs]
cmapdec6 = [3 for c in dec6cs]
cmap = mpl.colormaps['viridis'].resampled(4)

colors= cmap12+cmap6+cmapdec6
colors = cmap(colors)

pcares = pca.fit_transform(parameterarray)
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.scatter(pcares[:,0],pcares[:,1],pcares[:,2],c = colors,s = 30)
ax.view_init(elev=90., azim=0)
ax.set_zlim(-3,3)
ax.set_ylim(-5,6)
ax.set_xlim(-5,8)
plt.show()
#%%
initialcell =   0
initialnode = dec6_exp.lineagetrees[3]

def plotmaxprom(node,color,limitgen = 5):
    cell = node.cell
    print(cell.name)
    if not cell.outlier and cell.duration <90:
        if limitgen is not None and node.generation <=limitgen:
            times = [t for t in cell.times]
            values = [np.median(cell.ACsolidity[t]) if np.mean (cell.ACsolidity[t]) <0.5 else 0.3 for t in cell.times]
            try:
                plt.plot(times,values,c=color,alpha = 0.5)
            except:
                plt.plot(times,values,c='b',alpha=0.5)
                
    
    for n in node.children:
        plotmaxprom(n,color,limitgen)
for n in nov12_exp.lineagetrees[:]:
    
    plotmaxprom(n,'purple')

    
for n in nov6_exp.lineagetrees[:]:
    
    plotmaxprom(n,'r')


for n in dec6_exp.lineagetrees[:]:
    
    plotmaxprom(n,'b')
plt.show()            
    


#%%

for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <6:
    #     means =[(65535-c.mean[t])/10000 for t in c.times]
    #     meanint  =np.mean(means)
    #     plt.plot([t for t in c.times],[x-meanint for x in means])
    #     plt.scatter([t for t in c.times],[x-meanint for x in means])
        plt.plot([t-0 for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'r',alpha = 0.5)
        plt.scatter([t-0 for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'r',alpha = 0.5)



# for c in nov12_exp.cells:
#     if not c.outlier and c.node.generation == 4:
#     #     means =[(65535-c.mean[t])/10000 for t in c.times]
#     #     meanint  =np.mean(means)
#     #     plt.plot([t for t in c.times],[x-meanint for x in means])
#     #     plt.scatter([t for t in c.times],[x-meanint for x in means])
#         plt.plot([t-0 for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'b',alpha = 0.5)
#         plt.scatter([t-0 for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'b',alpha = 0.5)
        
for c in nov6_exp.cells:
    if not c.outlier and c.node.generation < 6:
    #     means =[(65535-c.mean[t])/10000 for t in c.times]
    #     meanint  =np.mean(means)
    #     plt.plot([t for t in c.times],[x-meanint for x in means])
    #     plt.scatter([t for t in c.times],[x-meanint for x in means])
        plt.plot([t-0 for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'purple',alpha = 0.5)
        plt.scatter([t-0 for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'purple',alpha = 0.5)


plt.show()

#%%
exp = nov12_exp
counter = 0
for c in exp.cells:
    if not c.outlier and c.node.generation <4 and counter <200:
        counter +=1
        plt.plot([t  for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'r')



exp = nov6_exp
counter = 0
for c in exp.cells:
    if not c.outlier and c.node.generation <4  and counter <200:
        counter +=1
        plt.plot([ t for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'purple')


exp = dec6_exp
counter = 0
for c in exp.cells:
    if not c.outlier and c.node.generation <4 and counter <200:
        counter +=1
        plt.plot([t  for t in c.times],[c.ACsolidity[t] for t in c.times],c = 'b')
plt.show()
#%%
# nov12_exp.plotaverage('mean',color = 'r')

# for c in nov12_exp.cells:
#     if c.node.generation <4 and not c.outlier:
#         plt.plot([t-c.times[0] for t in c.times],[c.mean[t]-nov12_exp.mean_avg[nov12_exp.times.index(t)] for t in c.times],c = 'r')
 
for c in nov6_exp.cells:
    if c.node.generation <4 and not c.outlier:
        plt.plot([t for t in c.times],[c.interior.mean[t]-nov6_exp.mean_avg[nov6_exp.times.index(t)] for t in c.times],c = 'b')
for c in dec6_exp.cells:
    if c.node.generation <4 and not c.outlier:
        plt.plot([t for t in c.times],[c.interior.mean[t]-dec6_exp.mean_avg[dec6_exp.times.index(t)] for t in c.times],c = 'purple')
#%%
for i in range(0,6):
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier:
            plt.plot([t-c.times[0]  for t in c.times],[c.interior.mean[t] for t in c.times],c = 'r',alpha = 0.4)
            
    for c in dec6_exp.cells:
        if c.node.generation == i and not c.outlier:
            plt.plot([t-c.times[0]  for t in c.times],[c.interior.mean[t] for t in c.times],c = 'b',alpha = 0.4)
    for c in nov12_exp.cells:
        if c.node.generation == i and not c.outlier:
            plt.plot([t-c.times[0]  for t in c.times],[c.interior.mean[t] for t in c.times],c = 'purple',alpha = 0.4)
            
    plt.title("Gen"+str(i))
    plt.show()
    #%%
def getsizerange(exp):
    exp.getglobalattr('area')
    maxes = [np.max(exp.area[t]) for t in exp.times]
    mins = [np.min(exp.area[t]) for t in exp.times]
    size_range = (np.min(mins),np.max(maxes))
    return size_range
nov6_range = getsizerange(nov6_exp)
nov12_range = getsizerange(nov12_exp)
dec6_range = getsizerange(dec6_exp)
nov6_exp.averagevalue('area')
nov12_exp.averagevalue('area')
dec6_exp.averagevalue('area')

def plotnvssize(exp,size_range,binsno=30,antibiotic = 1000):
    datalist = []
    fig,axs= plt.subplots(2,1)
    
    for t in exp.times:
        N = len(exp.area[t])
        avgsize = np.mean(exp.area[t])
        data = np.asarray(exp.area[t])/avgsize
        datalist.append(data)
        newrange = [size_range[i]/avgsize for i in range(2)]
        if t<antibiotic:
            c = 'cyan'
            bars,bins,_ = axs[0].hist(data,bins = binsno,histtype='bar',range = newrange,density=True,color = c)
            axs[0].set_title(exp.name+" pre-antibiotic")
            axs[0].set_xlabel("Size/Avg Size (Area)")
            axs[0].set_ylabel("Nbin/N")
            
            axs[0].set_ylim(0,12)
            axs[0].set_xlim(0,5)
            
            
    
        else:
            c = 'red'
            bars,bins,_ = axs[1].hist(data,bins = binsno,histtype='bar',range = newrange,density=True,color = c)
            axs[1].set_title(exp.name+" post-antibiotic")
            axs[1].set_xlabel("Size/Avg Size (Area)")
            axs[1].set_ylabel("Nbin/N")
            
            axs[1].set_ylim(0,12)
            axs[1].set_xlim(0,5)
        
        
    plt.show()
    colors = np.linspace(0,1,len(datalist))
    colors = plt.cm.viridis(colors)
    
    bars,bins,_ = plt.hist(datalist,bins = binsno,histtype='barstacked',range = newrange,density=True,color=colors)
    plt.title(exp.name+" stacked chart")
    plt.xlabel("Size/Avg Size (Area)")
    plt.ylabel("Nbin/N")
    
    plt.ylim(0,2)
    plt.xlim(0,5)
    plt.show()
plotnvssize(nov6_exp,nov6_range,antibiotic = 100000)
plotnvssize(dec6_exp,dec6_range,antibiotic = 100000)
plotnvssize(nov12_exp,nov12_range,antibiotic = 34)
#%%
for c in dec6_exp.cells:
    plt.plot(c.times,[c.mean[t] for t in c.times],c  = 'b')
    
for c in nov12_exp.cells:
    plt.plot(c.times,[c.mean[t]*8 for t in c.times],c  = 'r')
plt.show()
#%%
nov6_exp.getcellcount()
avgdurationgen = []
avgdurationstd = []
for i in range(1,7):
    tempd = []
    
    for c in nov6_exp.cells:
        if c.node.generation == i:
            tempd.append(c.duration)
    avgdurationgen.append(np.mean(tempd))
    avgdurationstd.append(np.std(tempd))
    
plt.errorbar(np.arange(1,7),avgdurationgen,yerr = avgdurationstd,c ='blue')
avgdurationgen = []
avgdurationstd = []
for i in range(0,5):
    tempd = []
    
    for c in nov12_exp.cells:
        if c.node.generation == i:
            tempd.append(c.duration)
    avgdurationgen.append(np.mean(tempd))
    avgdurationstd.append(np.std(tempd))
    
plt.errorbar(np.arange(0,5),avgdurationgen,yerr = avgdurationstd,c = 'red')
plt.show()
      


    
#%%

def plotpoles(node,select_children = None,fsize = 4,plot = True):
    sumlist = []
    
    c = node.cell
    divlist = [len(c.times)]
    for t in c.times:
        plt.figure(figsize = (fsize,fsize))
        if  c.polesprom[t] != [None]:
            polesc = deepcopy(c.polespromnorm)
            alpha = [x  for x in polesc[t]]
            alpha = np.nan_to_num(alpha)
            if plot:
                plt.scatter(c.polepositions[t][:,0],c.polesprom[t],alpha = alpha, c = plt.cm.viridis(polesc[t]),s=np.nan_to_num(c.polessizes[t])*(fsize**2))
                
                SPsum = np.sum(np.asarray(c.polesprom[t]))+np.sum(c.polessizes[t]/c.area[t])
                plt.ylim(0,8)
                plt.xlim((-40,40))
                plt.title(c.name + ":T = "+ str(t))
                plt.xlabel("Pozitia pe axa majora (centrata la 0)")
                plt.ylabel("Proeminenta (pol/fundal AC)")
                plt.show()
            
            SPsum =   len(c.polesprom[t])  #np.sum(np.nan_to_num(c.polessizes[t]))  # len(c.polessizes[t])
            sumlist.append(SPsum)
    if type(c.children) == list and c.children != []:
        if select_children == None:
             s,d = plotpoles(c.children[0].node,plot = plot)
             sumlist += s
             divlist += d
        elif type(select_children) == list:
            try:
                s,d =plotpoles(c.children[select_children[c.node.generation]].node,select_children = select_children,plot = plot)
                sumlist += s
                divlist += d
            except IndexError:
                print("Index out of range, defaulting to first branch")
                s,d = plotpoles(c.children[0].node,select_children = select_children,plot = plot)
                sumlist += s
                divlist += d
    return sumlist,divlist

    
ch = [1,1,0,0]
slist,divlist = plotpoles(dec6_exp.lineagetrees[2],select_children = ch,plot = False)     
plt.show()
plt.plot(dec6_exp.times,np.nan_to_num(slist))
plt.scatter(dec6_exp.times,np.nan_to_num(slist))
divlist = np.cumsum(divlist)
divlist = [dec6_exp.times[x-1] for x in divlist]
plt.vlines(x =divlist,ymin = 0,ymax = 10)
plt.xlim(0,divlist[2])
plt.ylabel("Distanta dintre polii extremi (px)")
plt.xlabel("T (min)")
plt.show()
#%%
decmeans = []
novmeans = []
times = [0,10,15,20,30]


majors = [(c.maxpoledistance[c.times[-1]]-c.maxpoledistance[c.times[0]])/(c.times[-1]-c.times[0]) for c in dec6_exp.cells if c.node.generation <3 and not c.outlier]
tpoints = [c.duration//5 for c in dec6_exp.cells if c.node.generation <3 and not c.outlier]
decmeans = [np.mean([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
destd = [np.std([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
dect = [i*5 +2.5 for i in range(1,6)]
tpoints = [t*5 +2.5 for t in tpoints]
print(decmeans)
print(destd)
plt.scatter(tpoints,majors,c = 'r',alpha = 0.3)

plt.errorbar(dect,decmeans,destd,c = 'r',ms = 10,fmt = 'o')
majors = [(c.maxpoledistance[c.times[-1]]-c.maxpoledistance[c.times[0]])/(c.times[-1]-c.times[0]) for c in nov6_exp.cells if c.node.generation <4 and c.node.generation >0 and not c.outlier]
tpoints = [c.duration//5 for c in nov6_exp.cells if c.node.generation <4 and c.node.generation >0 and not c.outlier]
decmeans = [np.mean([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
destd = [np.std([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
dect = [i*5 +2.5 for i in range(1,6)]
tpoints = [t*5 + 2.5 for t in tpoints]
print(decmeans)
print(destd)
plt.scatter(tpoints,majors,c = 'b',alpha = 0.3)

plt.errorbar(dect,decmeans,destd,c = 'b',ms = 10,fmt = 'o')
plt.xlim(0,35)
plt.grid(axis = 'x')
plt.xlabel("Durata (min)")
plt.ylabel("Panta de crestere (px/min)")
red_line  = mpl.lines.Line2D([], [], color='red', ls='-', markersize=8, label='Dec 6 control')
blue_line = mpl.lines.Line2D([], [], color='blue', ls='-', markersize=8, label='Nov 6 control')
plt.legend(handles=[red_line, blue_line])
#%%
decmeans = []
novmeans = []
times = [0,10,15,20,30]


majors = [c.majorslope for c in dec6_exp.cells if c.node.generation <3 and not c.outlier]
tpoints = [c.duration//5 for c in dec6_exp.cells if c.node.generation <3 and not c.outlier]
decmeans = [np.mean([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
destd = [np.std([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
dect = [i*5 +2.5 for i in range(1,6)]
tpoints = [t*5 +2.5 for t in tpoints]
print(decmeans)
print(destd)
plt.scatter(tpoints,majors,c = 'r',alpha = 0.3)

plt.errorbar(dect,decmeans,destd,c = 'r',ms = 10,fmt = 'o')
majors = [c.majorslope for c in nov6_exp.cells if c.node.generation <4 and c.node.generation >0 and not c.outlier]
tpoints = [c.duration//5 for c in nov6_exp.cells if c.node.generation <4 and c.node.generation >0 and not c.outlier]
decmeans = [np.mean([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
destd = [np.std([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
dect = [i*5 +2.5 for i in range(1,6)]
tpoints = [t*5 + 2.5 for t in tpoints]
print(decmeans)
print(destd)
plt.scatter(tpoints,majors,c = 'b',alpha = 0.3)

plt.errorbar(dect,decmeans,destd,c = 'b',ms = 10,fmt = 'o')
plt.xlim(0,35)
plt.grid(axis = 'x')
plt.xlabel("Durata (min)")
plt.ylabel("Panta de crestere (px/min)")
red_line  = mpl.lines.Line2D([], [], color='red', ls='-', markersize=8, label='Dec 6 control')
blue_line = mpl.lines.Line2D([], [], color='blue', ls='-', markersize=8, label='Nov 6 control')
plt.legend(handles=[red_line, blue_line])
#%%
decmeans = []
novmeans = []
times = [0,10,15,20,30]


majors = [ np.mean([[c.polesprom[t][i] for i in c.expoleindexes[t]] for t in c.times]) for c in dec6_exp.cells if c.node.generation <3 and not c.outlier]
tpoints = [c.duration//5 for c in dec6_exp.cells if c.node.generation <3 and not c.outlier]
decmeans = [np.mean([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
destd = [np.std([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
dect = [i*5 +2.5 for i in range(1,6)]
tpoints = [t*5 +2.5 for t in tpoints]
print(decmeans)
print(destd)
plt.scatter(tpoints,majors,c = 'r',alpha = 0.3)

plt.errorbar(dect,decmeans,destd,c = 'r',ms = 10,fmt = 'o')
majors = [np.mean([[c.polesprom[t][i] for i in c.expoleindexes[t]] for t in c.times]) for c in nov6_exp.cells if c.node.generation <4 and c.node.generation >0 and not c.outlier]
tpoints = [c.duration//5 for c in nov6_exp.cells if c.node.generation <4 and c.node.generation >0 and not c.outlier]
decmeans = [np.mean([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
destd = [np.std([x for x,y in zip(majors,tpoints) if y ==i]) for i in range(1,6)]
dect = [i*5 +2.5 for i in range(1,6)]
tpoints = [t*5 + 2.5 for t in tpoints]
print(decmeans)
print(destd)
plt.scatter(tpoints,majors,c = 'b',alpha = 0.3)

plt.errorbar(dect,decmeans,destd,c = 'b',ms = 10,fmt = 'o')
plt.xlim(0,35)
plt.grid(axis = 'x')
plt.xlabel("Durata (min)")
plt.ylabel("Media proeminentei polilor ")
red_line  = mpl.lines.Line2D([], [], color='red', ls='-', markersize=8, label='Dec 6 control')
blue_line = mpl.lines.Line2D([], [], color='blue', ls='-', markersize=8, label='Nov 6 control')
plt.legend(handles=[red_line, blue_line],fancybox = True)
#%%
for c in nov12_exp.cells:
    c.poleFFT(1)
    
for c in dec6_exp.cells:
    c.poleFFT(1)
peaks = []
exp = nov12_exp

for i in range(0,6):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    for c in dec6_exp.cells:
        if c.node.generation == i and not c.outlier and  'polesignal' in c.__dict__.keys() and len(c.times) > 3:
            promlist = [[c.polesprom[t][i] for i in c.expoleindexes[t]] for t in c.times]
            markers,bars,caps = ax1.errorbar([t-c.times[0] for t in c.times],[np.mean(x) for x in promlist],yerr = [np.std(x)/np.sqrt(len(x)) for x in promlist])
            ax1.plot(c.polesignal[0]/60,c.polesignal[1],c = markers.get_color())
            ax1.scatter([t-c.times[0] for t in c.times],[promlist[t][0] for t in range(len(c.times))],c = markers.get_color(),alpha = 0.2)
            ax1.scatter([t-c.times[0] for t in c.times],[promlist[t][1] for t in range(len(c.times))],c = markers.get_color(),alpha = 0.2)
            
            ax2.plot(c.expolefourier[0],c.expolefourier[1],c = markers.get_color())
            
            peakfreqs = [c.expolefourier[0][i] for i in find_peaks(c.expolefourier[1])[0]]
            
            peaks.append([x for x in peakfreqs if x < 0.004])
    #plt.ylim(0,10)
    
    ax1.set_title("Timp")
    ax1.set_xlabel("T (min)")
    ax1.set_ylabel("Proeminenta medie extreme")
    ax2.set_title("Frecventa")
    ax2.set_xlim(0,0.01)
    ax2.set_xlabel("Frecventa (Hz)")
    ax2.set_ylabel("Amplitudinea oscilatiilor (rms)")
    plt.show()
fullpeaks = []
for el in peaks:
    [fullpeaks.append(x) for x in el]
print(fullpeaks)
print(np.mean(fullpeaks))

#%%
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation ==3:
        plt.plot([t-c.times[0] for t in c.times],[c.mean[t] for t in c.times])


    

plt.show()
#%%
dec =[]
nov = []
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <3 and c.duration > 10 and c.duration < 25:
        value = np.mean([c.maxpoledistance[t]/c.major[t] for t in c.times])
        dec.append(value)
for c in nov6_exp.cells:
    if not c.outlier and c.node.generation <4 and c.node.generation >0 and c.duration > 10 and c.duration < 25:
        value = np.mean([c.maxpoledistance[t]/c.major[t] for t in c.times])
        nov.append(value)
total = dec+nov
data = plt.boxplot(total)  
plt.violinplot(total)  
plt.xticks([])
plt.ylabel("dPoli/axa majora")
print("Statistici")
print(np.mean(total))
print(np.std(total))
print(np.min(total))
print(np.max(total))
print(data["caps"][0].get_ydata())
print(data["boxes"][0].get_ydata)
print(data["caps"][1].get_ydata())
#%%
dec =[]
nov = []
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <3 and c.duration > 10 and c.duration < 25:
        value = c.majorslope
        dec.append(value)
for c in nov6_exp.cells:
    if not c.outlier and c.node.generation <4 and c.node.generation >0 and c.duration > 10 and c.duration < 25:
        value = c.majorslope
        nov.append(value)
total = dec+nov
data = plt.boxplot(total)
plt.violinplot(total)  
plt.xticks([])    
plt.ylabel(" Panta axei majore (px/min)")
print("Statistici")
print(np.mean(total))
print(np.std(total))
print(np.min(total))
print(np.max(total))
print(data["caps"][0].get_ydata())
print(data["boxes"][0].get_ydata)
print(data["caps"][1].get_ydata())
#%%
dec =[]
nov = []
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <3 and c.duration > 10 and c.duration < 25:
        value = (c.maxpoledistance[c.times[-1]]-c.maxpoledistance[c.times[0]])/(c.times[-1]-c.times[0])
        dec.append(value)
for c in nov6_exp.cells:
    if not c.outlier and c.node.generation <4 and c.node.generation >0 and c.duration > 10 and c.duration < 25:
        value = (c.maxpoledistance[c.times[-1]]-c.maxpoledistance[c.times[0]])/(c.times[-1]-c.times[0])
        nov.append(value)
total = dec+nov
data = plt.boxplot(total)  
plt.violinplot(total)  
plt.xticks([])  
plt.ylabel(" Panta distantei dintre poli (px/min)")
print("Statistici")
print(np.mean(total))
print(np.std(total))
print(np.min(total))
print(np.max(total))
print(data["caps"][0].get_ydata())
print(data["boxes"][0].get_ydata)
print(data["caps"][1].get_ydata())
#%%
dec =[]
nov = []
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <3 and c.duration > 10 and c.duration < 25:
        value = (c.maxpoledistance[c.times[-1]]-c.maxpoledistance[c.times[0]])/(c.times[-1]-c.times[0])/c.majorslope
        dec.append(value)
for c in nov6_exp.cells:
    if not c.outlier and c.node.generation <4 and c.node.generation >0 and c.duration > 10 and c.duration < 25:
        value = (c.maxpoledistance[c.times[-1]]-c.maxpoledistance[c.times[0]])/(c.times[-1]-c.times[0])/c.majorslope
        nov.append(value)
total = dec+nov
data = plt.boxplot(total) 
plt.violinplot(total)  
plt.xticks([])   
plt.ylabel(" dPoli/axa majora")
print("Statistici")
print(np.mean(total))
print(np.std(total))
print(np.min(total))
print(np.max(total))
print(data["caps"][0].get_ydata())
print(data["boxes"][0].get_ydata)
print(data["caps"][1].get_ydata())
#%%
dec =[]
nov = []
for c in nov12_exp.cells:
    if not c.outlier and c.node.generation <3 and c.duration > 10 and c.duration < 25:
        value = np.mean([c.ACsolidity[t] for t in c.times])
        dec.append(value)
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <4 and c.node.generation >0 and c.duration > 10 and c.duration < 25:
        value = np.mean([c.ACsolidity[t] for t in c.times])
        nov.append(value)
total = dec+nov
data = plt.boxplot(total)  
plt.violinplot(total)  
plt.xticks([])  
plt.ylabel("Media soliditatii AC")
print("Statistici")
print(np.mean(total))
print(np.std(total))
print(np.min(total))
print(np.max(total))
print(data["caps"][0].get_ydata())
print(data["boxes"][0].get_ydata)
print(data["caps"][1].get_ydata())
#%%
dec =[]
nov = []
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <3 and c.duration > 10 and c.duration < 25:
        expoles = [np.mean([c.polesprom[t][c.expoleindexes[t][0]],c.polesprom[t][c.expoleindexes[t][1]]]) for t in c.times]
        value = np.mean([np.mean([c.polesprom[t][c.expoleindexes[t][0]],c.polesprom[t][c.expoleindexes[t][1]]]) for t in c.times])
        dec.append(value)
for c in nov6_exp.cells:
    if not c.outlier and c.node.generation <4 and c.node.generation >0 and c.duration > 10 and c.duration < 25:
        value = np.mean([np.mean([c.polesprom[t][c.expoleindexes[t][0]],c.polesprom[t][c.expoleindexes[t][1]]]) for t in c.times])
        nov.append(value)
total = dec+nov
data = plt.boxplot(total)   
plt.violinplot(total)  
plt.xticks([]) 
plt.ylabel("Media proeminentei polilor AC")
print("Statistici")
print(np.mean(total))
print(np.std(total))
print(np.min(total))
print(np.max(total))
print(data["caps"][0].get_ydata())
print(data["boxes"][0].get_ydata)
print(data["caps"][1].get_ydata())
#%%
for c in dec6_exp.cells:
    if not c.outlier and c.node.generation <3:
        poledist = [c.maxpoledistance[t]/c.major[t] for t in c.times]
        trimmeddist = [x for x in poledist if x != 0]
       
        trimmedtimes = [x for x,y in zip(c.times,poledist) if y != 0]
        trimmedtimes = [x - trimmedtimes[0] for x in trimmedtimes]
        plt.plot(trimmedtimes,trimmeddist,c = 'r',alpha = 0.3)
means = [np.mean([c.maxpoledistance[t]/c.major[t] for t in c.times]) for c in dec6_exp.cells if c.node.generation < 3 and not c.outlier]
maxes = [np.max([c.maxpoledistance[t]/c.major[t]for t in c.times]) for c in dec6_exp.cells if c.node.generation < 3 and not c.outlier]
mins = [np.min([c.maxpoledistance[t]/c.major[t] for t in c.times]) for c in dec6_exp.cells if c.node.generation < 3 and not c.outlier]
print(np.max(maxes))
print(np.nanmin([x if x  != 0 else 100 for x in mins]))
print(np.mean(means))
print(np.std(means))
for c in nov6_exp.cells:
    if not c.outlier and c.node.generation <4 and c.node.generation >0:
        poledist = [c.maxpoledistance[t]/c.major[t] for t in c.times]
        trimmeddist = [x for x in poledist if x != 0]
        
        trimmedtimes = [x for x,y in zip(c.times,poledist) if y != 0]
        trimmedtimes = [x - trimmedtimes[0] for x in trimmedtimes]
        plt.plot(trimmedtimes,trimmeddist,c = 'b',alpha = 0.3)
        
plt.xlabel("T (min)")
plt.ylabel("dPoli/Axa majora")
plt.show()
means = [ np.mean([c.maxpoledistance[t]/c.major[t] for t in c.times]) for c in nov6_exp.cells if c.node.generation < 4 and not c.outlier and c.node.generation >0]
maxes = [ np.max([c.maxpoledistance[t]/c.major[t] for t in c.times]) for c in nov6_exp.cells if c.node.generation < 4 and not c.outlier and c.node.generation >0]
mins = [ np.min([c.maxpoledistance[t]/c.major[t] for t in c.times]) for c in nov6_exp.cells if c.node.generation < 4 and not c.outlier and c.node.generation >0]
print(np.max(maxes))
print(np.nanmin([x if x  != 0 else 100 for x in mins]))
print(np.mean(means))
print(np.std(means))
nov6_exp.plotaverage('mean')
dec6_exp.plotaverage('mean',color = 'r')
plt.xlabel("T(min)")
plt.ylabel("Intensitatea medie")
plt.show()

plt.errorbar(nov6_exp.times,nov6_exp.mean_avg,nov6_exp.mean_std,c = 'b')
plt.errorbar(dec6_exp.times,dec6_exp.mean_avg,dec6_exp.mean_std,c = 'r')
plt.xlabel("T(min)")
plt.ylabel("Intensitatea medie")
red_line  = mpl.lines.Line2D([], [], color='red', ls='-', markersize=8, label='Dec 6 control')
blue_line = mpl.lines.Line2D([], [], color='blue', ls='-', markersize=8, label='Nov 6 control')
plt.legend(handles=[red_line, blue_line],fancybox = True)
plt.show()

#%%
nov12_exp.plotaverage("maxpoledistance")
plt.show()
decline = []
novline = []

dectime = []
novtime = []

times = [0,10,15,20,30]


for c in dec6_exp.cells:
    if c.node.generation < 3 and not c.outlier :
        plt.plot([t-c.times[0] for t in c.times],[c.maxpoledistance[t]/c.major[t] for t in c.times],c = 'r',alpha = 0.3)
        decline += [c.major[t]/c.maxpoledistance[t] for t in c.times]
        dectime += [t-c.times[0] for t in c.times]
        lis = [c.area[t] for t in c.times]
       
slope, intercept, r_value, p_value, std_err = st.linregress(dectime, decline)
means = [np.mean([c.ACsolidity[t] for t in c.times]) for c in dec6_exp.cells if c.node.generation < 3 and not c.outlier]
maxes = [np.max([c.ACsolidity[t] for t in c.times]) for c in dec6_exp.cells if c.node.generation < 3 and not c.outlier]
mins = [np.min([c.ACsolidity[t] for t in c.times]) for c in dec6_exp.cells if c.node.generation < 3 and not c.outlier]
print('\n')
print("6 dec")
print(np.max(maxes))
print(np.min(mins))
print(np.mean(means))
print(np.std(means))
print(std_err)
print('\n')
time = np.linspace(0,np.max(dectime),1000)
plt.plot(time,slope*time+intercept,ls = '--',c = 'r')


for c in nov12_exp.cells:
    if c.node.generation < 4 and not c.outlier and c.node.generation >0 :
        plt.plot([t-c.times[0] for t in c.times],[c.maxpoledistance[t]/c.major[t] for t in c.times],c = 'b',alpha = 0.3)
        decline += [c.major[t]  for t in c.times]
        dectime += [t-c.times[0] for t in c.times]
        lis = [c.area[t] for t in c.times]
        
    
slope, intercept, r_value, p_value, std_err = st.linregress(dectime, decline)
means = [ np.mean([c.ACsolidity[t] for t in c.times]) for c in nov6_exp.cells if c.node.generation < 4 and not c.outlier and c.node.generation >0]
maxes = [ np.max([c.ACsolidity[t] for t in c.times]) for c in nov6_exp.cells if c.node.generation < 4 and not c.outlier and c.node.generation >0]
mins = [ np.min([c.ACsolidity[t] for t in c.times]) for c in nov6_exp.cells if c.node.generation < 4 and not c.outlier and c.node.generation >0]


print('\n')
print("6 nov")
print(np.max(maxes))
print(np.min(mins))
print(np.mean(means))
print(np.std(means))
print(std_err)
print('\n')
time = np.linspace(0,np.max(dectime),1000)
plt.plot(time,slope*time+intercept,ls = '--',c = 'b')

red_line  = mpl.lines.Line2D([], [], color='red', ls='-', markersize=8, label='Dec 6 control')
blue_line = mpl.lines.Line2D([], [], color='blue', ls='-', markersize=8, label='Nov 6 control')

red_linefit  = mpl.lines.Line2D([], [], color='red', ls='--', markersize=8, label='Best-fit: dec 6 control')
blue_linefit = mpl.lines.Line2D([], [], color='blue',ls='--', markersize=8, label='Best-fit: Nov 6 control')
plt.legend(handles=[red_line, blue_line,red_linefit, blue_linefit])
plt.xlabel("T (min)")
plt.ylabel("Distanta normalizata dintre poli (px)")
plt.title("Durata:"+str(times[i])+"-"+str(times[i+1]))
plt.show()
#%%
def calculateslope(cellist,value):
    values = []
    times = []
    try:
        for c in cellist:
            v = [c.__dict__[value][t] for t in c.times]
            v = [x-v[0] for x in v]
            t = [t-c.times[0] for t in c.times]
            values += v
            times += t
        slope, intercept, r_value, p_value, std_err = st.linregress(times, values)
        print(slope)
        print(r_value**2)
        print(p_value)
        print('\n')
    except ValueError:
        print("Value is not in celldict!")
    
cellist = [c for c in nov6_exp.cells if c.node.generation <4 and c.node.generation > 0 and not c.outlier]
calculateslope(cellist,'major')


cellist = [c for c in dec6_exp.cells if c.node.generation <3 and c.node.generation > -1 and not c.outlier]
calculateslope(cellist,'major')

#%%
for c in dec6_exp.cells:
    if c.node.generation < 3 and not c.outlier :
        
        if np.mean([c.polesprom[t][c.expoleindexes[t][0]] for t in c.times]) > np.mean([c.polesprom[t][c.expoleindexes[t][1]] for t in c.times]):
            expole = [c.polessizes[t][c.expoleindexes[t][0]] for t in c.times]
        else:
            
            expole = [c.polessizes[t][c.expoleindexes[t][0]] for t in c.times]
        
        if np.max(expole)< 100 and c.times[-1]-c.times[0] >0:
            plt.plot([t-c.times[0]-shift for t in c.times],expole,c = 'r')
            plt.scatter([t-c.times[0]-shift for t in c.times],expole,c = 'r')
            
for c in nov6_exp.cells:
    if c.node.generation < 4 and not c.outlier :
        if np.mean([c.polesprom[t][c.expoleindexes[t][0]] for t in c.times]) > np.mean([c.polesprom[t][c.expoleindexes[t][1]] for t in c.times]):
            expole = [c.polessizes[t][c.expoleindexes[t][0]] for t in c.times]
        else:
            
            expole = [c.polessizes[t][c.expoleindexes[t][0]] for t in c.times]
        
        if np.max(expole)< 100 and c.times[-1]-c.times[0] >0:
            plt.plot([t-c.times[0] for t in c.times],expole,c = 'b')
            plt.scatter([t-c.times[0] for t in c.times],expole,c = 'b')
        
for c in nov12_exp.cells:
    if  not c.outlier and c.times[-1] < 45:
        if np.mean([c.polesprom[t][c.expoleindexes[t][0]] for t in c.times]) > np.mean([c.polesprom[t][c.expoleindexes[t][1]] for t in c.times]):
            expole = [c.polessizes[t][c.expoleindexes[t][0]] for t in c.times]
        else:
            
            expole = [c.polessizes[t][c.expoleindexes[t][0]] for t in c.times]
        
        if np.max(expole)< 100 and c.times[-1]-c.times[0] >0:
            plt.plot([t-c.times[0] for t in c.times],expole,c = 'purple')
            plt.scatter([t-c.times[0] for t in c.times],expole,c = 'purple')
            
            
#%%
gens = []
genstd = []
for i in range(0,3):
    genav = []
    for c in dec6_exp.cells:
        if c.node.generation == i:
            genav += [c.ACsolidity[t] for t in c.times]
    gens.append(np.mean(genav)) 
    genstd.append(np.std(genav))
plt.errorbar(np.arange(1,4),gens,yerr = genstd,c = 'r')
slope, intercept, r_value, p_value, std_err = st.linregress(np.arange(1,4), gens)
plt.plot(np.arange(1,4),slope*np.arange(1,4)+intercept,ls = '--', c = 'r')
print('\n')
print("6 dec")
print(slope)
print(r_value**2)
print(std_err)
print('\n')
gens = []
genstd = []
for i in range(1,4):
    genav = []
    for c in nov12_exp.cells:
        if c.node.generation == i:
            genav += [c.ACsolidity[t] for t in c.times]
    gens.append(np.mean(genav)) 
    genstd.append(np.std(genav))
plt.errorbar(np.arange(1,4),gens,yerr = genstd,c = 'b')
slope, intercept, r_value, p_value, std_err = st.linregress(np.arange(1,4), gens)
plt.plot(np.arange(1,4),slope*np.arange(1,4)+intercept,ls = '--', c = 'b')
print('\n')
print("6 nov")

print(slope)
print(r_value**2)
print(std_err)
print('\n')

red_line  = mpl.lines.Line2D([], [], color='red', ls='-', markersize=8, label='Dec 6 control')
blue_line = mpl.lines.Line2D([], [], color='blue', ls='-', markersize=8, label='Nov 6 control')

red_linefit  = mpl.lines.Line2D([], [], color='red', ls='--', markersize=8, label='Best-fit: dec 6 control')
blue_linefit = mpl.lines.Line2D([], [], color='blue',ls='--', markersize=8, label='Best-fit: Nov 6 control')
plt.legend(handles=[red_line, blue_line,red_linefit, blue_linefit])
plt.xlabel("Generatii")
plt.ylabel("Soliditate (Suprafata AC/Suprafata DC)")
plt.show()
#%%
# Durata CC
mean = []
for c in dec6_exp.cells:
    if c.node.generation <3 and not c.outlier:
        plt.scatter(c.node.generation+1,c.duration,c = 'r')
        mean.append(c.times[-1]-c.times[0])
for c in nov6_exp.cells:
    if c.node.generation <4 and not c.outlier and c.node.generation >0:
        plt.scatter(c.node.generation,c.duration,c = 'b')
        mean.append(c.times[-1]-c.times[0])
red_line  = mpl.lines.Line2D([], [], color='red', marker='o', markersize=8, label='Dec 6 control')
blue_line = mpl.lines.Line2D([], [], color='blue', marker='o', markersize=8, label='Nov 6 control')
plt.legend(handles=[red_line, blue_line])
plt.xlabel("Generatii")
plt.ylabel("Durata (min)")
plt.show()
print(np.mean(mean))
print(np.std(mean))
print(np.max(mean))
print(np.min(mean))
    
#%%
# for i in range(0,3):
#     for c in dec6_exp.cells:
#         if c.node.generation == i and not c.outlier:
#             promlist = [[c.polesprom[t][i] for i in range(len(c.polesprom[t])) if i not in c.expoleindexes[t]]for t in c.times]
#             plt.errorbar(c.times,[np.mean(x) for x in promlist],yerr = [np.std(x)/np.sqrt(len(x)) for x in promlist])
            
#     #plt.ylim(0,10)
#     plt.title("Gen "+str(i))
#     plt.xlabel("T")
#     plt.ylabel("Proeminenta medie centru")
    
#     plt.show()
    


# for i in range(0,3):
#     for c in dec6_exp.cells:
#         if c.node.generation == i and not c.outlier:
#             plt.plot(c.times,[c.interior.mean[t] for t in c.times])
            
#     #plt.ylim(0,10)
#     plt.title("Gen "+str(i))
#     plt.xlabel("T")
#     plt.ylabel("Intensitatea medie AC")
    
#     plt.show()


ch = [1,1,0,0]
slist,divlist = plotpoles(nov6_exp.lineagetrees[0],select_children = ch,plot = False)     
plt.show()
plt.plot(nov6_exp.times,np.nan_to_num(slist))
plt.scatter(nov6_exp.times,np.nan_to_num(slist))
divlist = np.cumsum(divlist)
divlist = [nov6_exp.times[x-1] for x in divlist]
plt.vlines(x =divlist,ymin = 0,ymax = 10)
plt.xlim(divlist[0],divlist[3])
plt.ylabel("Distanta dintre polii extremi (px)")
plt.xlabel("T (min)")
plt.show()

# for i in range(1,4):
#     for c in nov6_exp.cells:
#         if c.node.generation == i and not c.outlier:
#             promlist = [[c.polesprom[t][i] for i in c.expoleindexes[t]] for t in c.times]
#             markers,bars,caps = plt.errorbar(c.times,[np.mean(x) for x in promlist],yerr = [np.std(x)/np.sqrt(len(x)) for x in promlist])
#             plt.scatter(c.times,[promlist[t][0] for t in range(len(c.times))],c = markers.get_color(),alpha = 0.2)
#             plt.scatter(c.times,[promlist[t][1] for t in range(len(c.times))],c = markers.get_color(),alpha = 0.2)
#     #plt.ylim(0,10)
#     plt.title("Gen "+str(i))
#     plt.xlabel("T")
#     plt.ylabel("Proeminenta medie extreme")
    
#     plt.show()      
# outlist = []
# for c in dec6_exp.cells:
#     if c.times[-1]-c.times[0] <8 and c.node.generation == 1:
#         outlist.append(c)
        
# for i in range(1,4):
#     for c in nov6_exp.cells:
#         if c.node.generation == i and not c.outlier:
#             promlist = [[c.polesprom[t][i] for i in range(len(c.polesprom[t])) if i not in c.expoleindexes[t]]for t in c.times]
#             plt.errorbar(c.times,[np.mean(x) for x in promlist],yerr = [np.std(x)/np.sqrt(len(x)) for x in promlist])
            
#     #plt.ylim(0,10)
#     plt.title("Gen "+str(i))
#     plt.xlabel("T")
#     plt.ylabel("Proeminenta medie centru")
    
#     plt.show() 

# for i in range(1,4):
#     for c in nov6_exp.cells:
#         if c.node.generation == i and not c.outlier:
#             plt.plot(c.times,[c.interior.mean[t] for t in c.times])
            
#     #plt.ylim(0,10)
#     plt.title("Gen "+str(i))
#     plt.xlabel("T")
#     plt.ylabel("Intensitatea medie AC")
    
#     plt.show
gens = [0]
for g in gens:
    for c in dec6_exp.cells:
        if 'expolefourier' in c.__dict__.keys() and c.node.generation == g: 
            plt.plot(c.expolefourier[0],c.expolefourier[1])
            plt.title(c.name)
plt.title("Spectru Fourier generatia "+str(g))
plt.xlabel("Frecventa (Hz)")
plt.ylabel("amplitudinea (rms^2, unitati de proeminenta)")
plt.show()


    #%%

for i in range(1,6):
    h = []
    hi = []
    b = []
    bi = []
    hg = []
    bg = []
    for c in nov12_exp.cells:
        if i == c.node.generation:
            for t in c.contour.times:
                h.append(c.contour.histogram[t])
                hi.append(c.interior.histogram[t])
                hg.append(c.histogram[t])
                b.append(c.contour.bins[t])
                bi.append(c.interior.bins[t])
                bg.append(c.bins[t])
        
    h = np.mean(h,axis = 0)
    b = np.mean(b,axis = 0)
    hi = np.mean(hi,axis = 0)
    bi = np.mean(bi,axis = 0)
    hg = np.mean(hg,axis = 0)
    bg = np.mean(bg,axis = 0)
    plt.plot(b,h,label = "Contur")
    plt.plot(bi,hi,label = "Interior")
    plt.plot(bg,hg,label = "Total")
    # plt.xlim(0,10)
    plt.xlim(5000,20000)
    plt.legend()
    plt.title("Antibiotic 10 mgL gen "+str(i))
    plt.show()
# nov12_names = ['Track_109.aaa','Track_41.bbb','Track_109.aaab','Track_41.bbba','Track_41.bbbaa','Track_109.aaabb']
# for t in nov12_exp.times:
#     counter = 0
#     for c in nov12_exp.cells:
        
#         if t in c.times:
#             plt.plot(c.interior.bins[t],c.interior.histogram[t],c = 'b',label = 'Interior')
#             plt.plot(c.contour.bins[t],c.contour.histogram[t],c = 'orange',label = 'Contour')
#             plt.plot(c.bins[t],c.histogram[t],c = 'g',label = 'Total')
#             plt.title(str(t)+"-sol = "+str(c.solidity[t]))
#             plt.legend()
#             plt.ylim(0,0.15)
#             counter+=1
#         if counter>1:
#             plt.show()
#             break
nov12_names = ['Track_109.a']

def plothistlin(exp,originalcells,endx = 30000):
    c = originalcells
    
    for t in exp.times:
            if t in c.times:
                plt.plot(c.interior.bins[t],c.interior.histogram[t],c = 'b',label = 'Interior ' +str(c.name))
                plt.plot(c.contour.bins[t],c.contour.histogram[t],c = 'orange',label = 'Contour '+str(c.name))
                plt.plot(c.bins[t],c.histogram[t],c = 'g',label = 'Total '+str(c.name))
                plt.title('T ='+str(t)+"-sol = "+str(c.solidity[t]))
                plt.legend()
                plt.show()

    
    if c.children == []:
            return
    plothistlin(exp,c.children[0],endx)
# plothistlin(nov12_exp,[c for c in nov12_exp.cells if c.name == 'Track_109.ab'][0])       
# plothistlin(nov6_exp,[c for c in nov6_exp.cells if c.name == 'Track_2.aa'][0])   
# plothistlin(nov5_exp,[c for c in nov5_exp.cells if c.name == 'Track_11.a'][0])             
# def plothistogramgens(exp,gen = 3,ncells = 2):
#     counter = 0
#     clist=[]
#     for c in exp.cells:
#         if c.node.generation == gen and c not in clist:
#             clist.append(c)
#             counter +=1
#         if counter == ncells:
#             break
    
#     newcellist = []
#     for c in clist:
#             for t in c.times:
#             plt.plot(c.interior.bins[t],c.interior.histogram[t],c = 'b',label = 'Interior'+c.name)
#             plt.plot(c.contour.bins[t],c.contour.histogram[t],c = 'orange',label = 'Contour'+c.name)
#             plt.plot(c.bins[t],c.histogram[t],c = 'g',label = 'Total'+c.name)
#             plt.title(str(t)+"-sol = "+str(c.solidity[t]))
#             plt.legend()
#             plt.ylim(0,0.15)
#             plt.xlim(0,30000)
#         if c.children == None:
#             continue
#         newcellist.append(c.children[0])
#     plt.show()
#     if newcellist != []:
#         plothistogramgens(newcellist,gen+1,ncells)
     
# for t in nov6_exp.times:
#     counter = 0
#     for c in nov6_exp.cells:
#         if t in c.times and c.node.generation!=0:
#             plt.plot(c.interior.bins[t],c.interior.histogram[t],c = 'b',label = 'Interior')
#             plt.plot(c.contour.bins[t],c.contour.histogram[t],c = 'orange',label = 'Contour')
#             plt.plot(c.bins[t],c.histogram[t],c = 'g',label = 'Total')
#             plt.title(str(t)+"-sol = "+str(c.ACsolidity[t]))
#             plt.legend()
#             plt.ylim(0,0.15)
#             plt.xlim(0,7)
#             counter+=1
#         if counter>1:
#             plt.show()
#             break


for i in range(1,6):
    h = []
    hi = []
    b = []
    bi = []
    bg = []
    hg = []
    for c in dec6_exp.cells:
        if i == c.node.generation:
            for t in c.contour.times:
                h.append(c.contour.histogram[t])
                hi.append(c.interior.histogram[t])
                hg.append(c.histogram[t])
                b.append(c.contour.bins[t])
                bi.append(c.interior.bins[t])
                bg.append(c.bins[t])
    h = np.mean(h,axis = 0)
    b = np.mean(b,axis = 0)
    hi = np.mean(hi,axis = 0)
    bi = np.mean(bi,axis = 0)
    hg = np.mean(hg,axis = 0)
    bg = np.mean(bg,axis = 0)
    plt.plot(b,h,label = "Contur")
    plt.plot(bi,hi,label = "Interior")
    plt.plot(bg,hg,label = "Total")

    plt.xlim(5000,20000)

    plt.title("Control gen "+str(i))
    plt.legend()
    plt.show()
    
for i in range(1,6):
    h = []
    hi = []
    b = []
    bi = []
    bg = []
    hg = []
    for c in nov6_exp.cells:
        if i == c.node.generation:
            for t in c.contour.times:
                h.append(c.contour.histogram[t])
                hi.append(c.interior.histogram[t])
                hg.append(c.histogram[t])
                b.append(c.contour.bins[t])
                bi.append(c.interior.bins[t])
                bg.append(c.bins[t])
    h = np.mean(h,axis = 0)
    b = np.mean(b,axis = 0)
    hi = np.mean(hi,axis = 0)
    bi = np.mean(bi,axis = 0)
    hg = np.mean(hg,axis = 0)
    bg = np.mean(bg,axis = 0)
    plt.plot(b,h,label = "Contur")
    plt.plot(bi,hi,label = "Interior")
    plt.plot(bg,hg,label = "Total")

    plt.xlim(5000,20000)

    plt.title("Control gen "+str(i))
    plt.legend()
    plt.show()


# for i in range(1,5):
#     h = []
#     hi = []
#     b = []
#     bi = []
#     bg = []
#     hg = []
#     for c in nov12_exp.cells:
#         if i == c.node.generation:
#             for t in c.contour.times:
#                 h.append(c.contour.histogram[t])
#                 hi.append(c.interior.histogram[t])
#                 hg.append(c.histogram[t])
#                 b.append(c.contour.bins[t])
#                 bi.append(c.interior.bins[t])
#                 bg.append(c.bins[t])
#     h = np.mean(h,axis = 0)
#     b = np.mean(b,axis = 0)
#     hi = np.mean(hi,axis = 0)
#     bi = np.mean(bi,axis = 0)
#     hg = np.mean(hg,axis = 0)
#     bg = np.mean(bg,axis = 0)
#     plt.plot(b,h,label = "Contur")
#     plt.plot(bi,hi,label = "Interior")
#     plt.plot(bg,hg,label = "Total")
#     plt.ylim(0,0.15)
    
#     plt.title("Antibiotic 5 mgL gen "+str(i))
#     plt.legend()
#     plt.show()
# for c in nov12_exp.cells:
#     plt.scatter(c.times,[c.contour.area[t]/c.interior.area[t] for t in c.times])
# plt.show()

# for c in nov12_exp.cells:
#     plt.scatter(c.times,[c.major[t] for t in c.times])
# plt.show()
# affectedcells = []
# healthycells = []
# for t in nov12_exp.times[25:]:
#     for c in nov12_exp.cells:
#         if t in c.times and c not in affectedcells and c not in healthycells:
#             if c.major[c.times[-1]] - c.major[c.times[0]] < c.major[t]*0.1 and c.times[0] != c.times[-1]:
#                 affectedcells.append(c)
#             else:
#                 healthycells.append(c)

# c = nov12_exp.cells[122]
# for t in c.times:
#     plt.plot(c.interior.bins[t],c.interior.histogram[t],c = 'b')
#     plt.plot(c.contour.bins[t],c.contour.histogram[t],c = 'red')
#     plt.title(t)
# plt.show()
# print(len(affectedcells))
# print(len(healthycells))
# for c in affectedcells:
#     plt.plot(c.times,[c.mean[t]/c.contour.mean[t] for t in c.times])
# nov6_exp.plotaverage('solidity')
# plt.show()
# for c in nov12_exp.cells:
#     plt.plot(c.times,[c.interior.mean[t]/c.contour.mean[t] for t in c.times],c = 'purple')
# # for c in nov6_exp.cells:
# #     plt.plot(c.times,[c.interior.mean[t]/c.contour.mean[t] for t in c.times],c = 'r')
# plt.show()

# for t in nov12_exp.times[18:32]:
#     for c in nov12_exp.cells:
#         if t in c.times:
#             plt.plot(c.contour.bins[t],c.contour.histogram[t],label = c.name)
    
#     plt.legend()
#     plt.title(t)
#     plt.show()
            
# for c in nov12_exp.cells:
#     plt.scatter(c.times,[c.contour.max[t]-c.contour.min[t] for t in c.times])
    
# plt.show()
# print(c.contour.bins[0])
# cs = ['b','r','g','orange','cyan','magenta','black']
# for i in range(len(c.times)):
#     plt.plot(c.contour.bins[i],[x/c.contour.area[i] for x in c.contour.histogram[i]],c = plt.cm.magma(i/len(c.times)),alpha  = 0.3)
#     plt.xlim(0,35000)
#     plt.show()

dec6_exp.plotaverage('intcontoverlap',color = 'r')

nov12_exp.plotaverage('intcontoverlap',color = 'purple')
plt.show()
# nov5_exp.plotaverage('intcontoverlap',color = 'g')
# plt.show()

# nov6_exp.plotaverage('intcontoverlap',color = 'r')

# nov12_exp.plotaverage('intcontoverlap',color = 'purple')

# nov5_exp.plotaverage('intcontoverlap',color = 'g')
# plt.show()
def plotaveragemaxAC(exp,color = 'b'):
    avg = []
    std = []
    for t in exp.times:
        tvals = []
        for c in exp.cells:
            if t in c.times:
                index = np.argwhere(np.asarray(c.histogram[t])>0)[-5:-1]
                value = np.mean([c.bins[t][i] for i in index])
                tvals.append(value)
        avg.append(np.mean(tvals))
        std.append(np.std(tvals))
    markers,bars,caps = plt.errorbar(exp.times,avg,yerr = std,fmt = 'o',c = color)
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]
    
plotaveragemaxAC(nov6_exp,color = 'r')
plotaveragemaxAC(nov12_exp,color = 'purple')



#clist = [c for c in nov6_exp.cells if c.duration > 30]
# nov1_exp.plotaverage('cellcount',color = 'orange')
# sept6_exp.plotaverage('cellcount',color = 'b')
# sept11_exp.plotaverage('cellcount',color = 'g')
# sept13_exp.plotaverage('cellcount',color = 'r')
# sept12_exp.plotaverage('cellcount',color = 'k')
# sept122_exp.plotaverage('cellcount',color = 'k')
# nov6_exp.plotaverage('cellcount',color = 'k')
# nov4_exp1.plotaverage('cellcount',color = 'k')
# nov4_exp2.plotaverage('cellcount',color = 'k')
# nov5_exp.plotaverage('cellcount',color = 'purple')
# oct11_exp.plotaverage('cellcount',color = 'cyan')
# oct15_exp.plotaverage('cellcount',color = 'magenta')

#%% 

def moments(x,y,index = 0):
    mean = np.sum((x)*y)
    std = np.sqrt(np.sum((x-mean)**2*y))
    skew = np.sum([(x-mean)**3*y])/(std**3)
    kurt = np.sum([(x-mean)**4*y])/(std**4)
    momlist = [mean,std,skew,kurt]
    return momlist[index]

def plotintcellratio(exp,moment = 1,color = 'b',batch = False,tshift = 0):
    
    for t in exp.times:
        avg = []
        std = []
        for c in exp.cells:
            if t in c.times:
                comint = moments(c.interior.bins[t],c.interior.histogram[t],index = moment-1)
                comcon = moments(c.contour.bins[t],c.contour.histogram[t],index = moment-1)
                comcell = moments(c.bins[t],c.histogram[t],index = moment-1)
                avg.append(comint/comcell)
                std.append(comint/comcell)
        avg = np.mean(avg)
        std = np.std(std)
        plt.errorbar(t+tshift,avg,yerr = 0,c = color,fmt = 'o')
    if not batch:
        plt.show()
colorlist = ['r','purple','green']
tshifs = [0,30,60]
explist = [nov12_exp,dec6_exp]
for e in explist:
    plotintcellratio(e,moment = 1,color = colorlist[explist.index(e)],batch = True,tshift = tshifs[explist.index(e)])
plt.show()
#%%

# for e in explist:
#     e.resetsolarea(1,3)
   
#     e.getglobalattr('solarea')
#     e.averagevalue('solarea')
#     plt.errorbar([t+tshifs[explist.index(e)] for t in e.times],e.solarea_avg,yerr = e.solarea_std,c = colorlist[explist.index(e)],fmt = 'o')
# plt.show()   
# for e in explist:
#     e.resetsolarea(1,3)
#     for c in e.cells:
#         c.plotvalue('solarea',t_shift = tshifs[explist.index(e)])
#         plt.ylim(0.2,0.7)
#     plt.title(e.name)
#     plt.show()
                         
# plt.show()
nov6_exp.resetsolarea(0.4,0.5)
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            if c.colornorm[t] <0.5:
                avgc.append(c.solarea[t]*c.area[t])
                stdc.append(c.solarea[t]*c.area[t])
            else:
                avgp.append(c.solarea[t]*c.area[t])
                
                stdp.append(c.solarea[t]*c.area[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
# plt.ylim(0.0,0.1)
plt.show()
limitlist = [[0.1,0.2],[0.1,0.4],[0.2,0.5],[0,0.8]]
for lim in limitlist:
    nov6_exp.resetsolarea(lim[0],lim[1])
    for t in nov6_exp.times:
         
        avgc = []
        stdc = []
        for c in nov6_exp.cells:
            if t in c.times:
                
                avgc.append(c.solarea[t])
                stdc.append(c.solarea[t])
                
        avgc = np.mean(avgc)
        stdc = np.std(stdc)
        plt.errorbar(t,avgc,yerr = stdc,c = 'k',fmt = 'o')

    plt.ylabel("Solid area/Total area")
    plt.xlabel("T (min)")
    plt.title("Soliditate vs. timp pentru intensitati intre "+str(lim))
    plt.show()

for lim in limitlist:
    nov6_exp.resetsolarea(lim[0],lim[1])
    for t in nov6_exp.times:
         
        avgc = []
        stdc = []
        avgp = []
        stdp = []
        for c in nov6_exp.cells:
            if t in c.times:
                
                if c.colornorm[t] <0.5:
                    avgc.append(c.solarea[t])
                    stdc.append(c.solarea[t])
                else:
                    avgp.append(c.solarea[t])
                    
                    stdp.append(c.solarea[t])
        avgc = np.mean(avgc)
        avgp = np.mean(avgp)
        stdc = np.std(stdc)
        stdp = np.std(stdp)
        plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
        plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')

    plt.ylabel("Solid area/Total area")
    plt.xlabel("T (min)")
    plt.title("Soliditate vs. timp pentru intensitati intre "+str(lim))
    plt.show()
    
for lim in limitlist:
    nov6_exp.resetsolarea(lim[0],lim[1])
    totalavg = []
    totalstd = []
    for t in nov6_exp.times:
         
        avgc = []
        stdc = []
        for c in nov6_exp.cells:
            if t in c.times:
                
                avgc.append(c.solarea[t])
                stdc.append(c.solarea[t])
                
        avgc = np.mean(avgc)
        stdc = np.std(stdc)
        totalavg.append(avgc)
        totalstd.append(stdc)
    plt.errorbar(nov6_exp.times,totalavg,yerr = totalstd,fmt = 'o',label = str(lim))

plt.legend()

plt.ylabel("px^2")
plt.xlabel("T (min)")
plt.title("Soliditate vs. timp suprapuse")
plt.show()     
limitlist = [[1,4],[0,1]]
acc = []
for lim in limitlist:
    nov6_exp.resetsolarea(lim[0],lim[1])
    totalavg = []
    totalstd = []
    for t in nov6_exp.times:
         
        avgc = []
        stdc = []
        for c in nov6_exp.cells:
            if t in c.times:
                
                avgc.append(c.solarea[t])
                stdc.append(c.solarea[t])
                
        avgc = np.mean(avgc)
        stdc = np.std(stdc)
        totalavg.append(avgc)
        totalstd.append(stdc)
        acc.append(totalavg)
    plt.errorbar(nov6_exp.times,totalavg,yerr = totalstd,fmt = 'o',label = str(lim))
plt.scatter(nov6_exp.times,[x+y for x,y in zip(acc[0],acc[1])], label = "[1,4]+[0,1]")
plt.legend()

plt.ylabel("PX^2")
plt.xlabel("T (min)")
plt.title("Soliditate vs. timp suprapuse")
plt.show()   
#%%
for c in nov6_exp.cells:
    plt.scatter(c.interior.times,[c.interior.histend[t] for t in c.interior.times],c = 'r')
for c in nov5_exp.cells:
    plt.scatter(c.interior.times,[c.interior.histend[t] for t in c.interior.times],c = 'g')

        
  
plt.show()      
#%%
# for c in nov6_exp.cells:
#     plt.scatter(c.times,[(c.interior.area[t]/c.area[t]) for t in c.times],c = 'r',alpha = 0.3)
# for c in nov5_exp.cells:
#     plt.scatter(c.times,[(c.interior.area[t]/c.area[t]) for t in c.times],c = 'g',alpha = 0.3)
for c in nov12_exp.cells:
    plt.scatter([t+20 for t in c.times],[(c.interior.area[t]/c.area[t]) for t in c.times],c = 'purple',alpha = 0.3)
for c in dec6_exp.cells:
    plt.scatter([t+20 for t in c.times],[(c.interior.area[t]/c.area[t]) for t in c.times],c = 'red',alpha = 0.3)
plt.show()    

#%%


nov6_exp.plotaverage('area',color = 'r')
plt.title("Suprafata DC mediata per moment de timp")
plt.xlabel("T (min)")
plt.ylabel("px^2")
plt.show()

nov6_exp.plotaverage('major',color = 'r')
plt.title("Axa majora DC mediata per moment de timp")
plt.xlabel("T (min)")
plt.ylabel("px")
plt.show()

nov6_exp.plotaverage('ar',color = 'r')
plt.title("Aspect ratio DC mediat per moment de timp")
plt.xlabel("T (min)")
plt.ylabel("Maj/Min")
plt.show()

avgs = []
stds = []
for t in nov6_exp.times:
    acc = []
    for c in nov6_exp.cells:
        
        if t in c.times:
            acc.append(c.interior.area[t])
    avgs.append(np.mean(acc))
    stds.append(np.std(acc))
markers,bars,caps = plt.errorbar(nov6_exp.times,avgs,yerr = stds,c = 'r',fmt = 'o')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]
plt.title("Suprafata AC mediata per moment de timp")
plt.xlabel("T (min)")
plt.ylabel("px^2")
plt.show()

avgs = []
stds = []
for t in nov6_exp.times:
    acc = []
    for c in nov6_exp.cells:
        
        if t in c.times:
            acc.append(c.interior.major[t])
    
    avgs.append(np.mean(acc))
    stds.append(np.std(acc))
markers,bars,caps = plt.errorbar(nov6_exp.times,avgs,yerr = stds,c = 'r',fmt = 'o')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]
plt.title("Axa majora AC mediata per moment de timp")
plt.xlabel("T (min)")
plt.ylabel("px")
plt.show()

avgs = []
stds = []
for t in nov6_exp.times:
    acc = []
    for c in nov6_exp.cells:
        
        if t in c.times:
            acc.append(c.interior.ar[t])
        
    avgs.append(np.mean(acc))
    stds.append(np.std(acc))
markers,bars,caps = plt.errorbar(nov6_exp.times,avgs,yerr = stds,c = 'r',fmt = 'o')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]

plt.title("Aspect ratio AC mediat per moment de timp")
plt.xlabel("T (min)")
plt.ylabel("Maj/Min")
plt.show()

#%%
avgs = []
stds = []
for t in dec6_exp.times:
    acc = []
    for c in dec6_exp.cells:
        
        if t in c.times:
            acc.append(c.ACsolidity[t])
        
    avgs.append(np.mean(acc))
    stds.append(np.std(acc))
markers,bars,caps = plt.errorbar([t+20 for t in dec6_exp.times],avgs,yerr = stds,c = 'k',fmt = 'o',label = 'AC')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]

avgs = []
stds = []
for t in nov12_exp.times:
    acc = []
    for c in nov12_exp.cells:
        
        if t in c.times:
            acc.append(c.ACsolidity[t])
        
    avgs.append(np.mean(acc))
    stds.append(np.std(acc))
markers,bars,caps = plt.errorbar(nov12_exp.times,avgs,yerr = stds,c = 'r',fmt = 'o',label = 'AC')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]


avgs = []
stds = []
for t in nov5_exp.times:
    acc = []
    for c in nov5_exp.cells:
        
        if t in c.times:
            acc.append(c.ACsolidity[t])
        
    avgs.append(np.mean(acc))
    stds.append(np.std(acc))
markers,bars,caps = plt.errorbar([t+20 for t in nov5_exp.times],avgs,yerr = stds,c = 'purple',fmt = 'o',label = 'AC')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]
plt.show()

#%%
for c in nov6_exp.cells:
    if c.duration >=10:
        plt.errorbar(c.duration,np.mean([c.area[t] for t in c.times]),yerr = np.std([c.area[t] for t in c.times]),c = plt.cm.magma(c.times[0]),fmt = 'o')
    
plt.title('Suprafata DC  medie per celula vs durata estimata ciclului celular')
plt.xlabel("Durata (min)")
plt.ylabel("px^2")
plt.show()

for c in nov6_exp.cells:
    if c.duration >=10:
        plt.errorbar(c.duration,np.mean([c.major[t] for t in c.times]),yerr = np.std([c.major[t] for t in c.times]),c = plt.cm.magma(c.times[0]),fmt = 'o')
    
plt.title('Axa majora DC medie per celula vs durata estimata ciclului celular')
plt.xlabel("Durata (min)")
plt.ylabel("px")
plt.show()

for c in nov6_exp.cells:
    if c.duration >=10:
        plt.errorbar(c.duration,np.mean([c.ar[t] for t in c.times]),yerr = np.std([c.ar[t] for t in c.times]),c = plt.cm.magma(c.times[0]),fmt = 'o')
    
plt.title('Aspect ratio DC mediu per celula vs durata estimata ciclului celular')
plt.xlabel("Durata (min)")
plt.ylabel("Maj/min")
plt.show()

for c in nov6_exp.cells:
    if c.duration >=10:
        plt.errorbar(c.duration,np.mean([c.interior.area[t] for t in c.times]),yerr = np.std([c.area[t] for t in c.times]),c = plt.cm.magma(c.times[0]),fmt = 'o')
    
plt.title('Suprafata AC  medie per celula vs durata estimata ciclului celular')
plt.xlabel("Durata (min)")
plt.ylabel("px^2")
plt.show()

for c in nov6_exp.cells:
    if c.duration >=10:
        plt.errorbar(c.duration,np.mean([c.interior.major[t] for t in c.times]),yerr = np.std([c.major[t] for t in c.times]),c = plt.cm.magma(c.times[0]),fmt = 'o')
    
plt.title('Axa majora AC medie per celula vs durata estimata ciclului celular')
plt.xlabel("Durata (min)")
plt.ylabel("px")
plt.show()

for c in nov6_exp.cells:
    if c.duration >=10:
        plt.errorbar(c.duration,np.mean([c.interior.ar[t] for t in c.times]),yerr = np.std([c.ar[t] for t in c.times]),c = plt.cm.magma(c.times[0]),fmt = 'o')
    
plt.title('Aspect ratio AC mediu per celula vs durata estimata ciclului celular')
plt.xlabel("Durata (min)")
plt.ylabel("Maj/min")
plt.show()
#%%
avgdiff = []    
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            
            if c.colornorm[t] <0.5:
                avgc.append(c.area[t])
                stdc.append(c.area[t])
            else:
                avgp.append(c.area[t])
                
                stdp.append(c.area[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
   
    avgdiff.append((avgp-avgc)/avgc)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
    
plt.title('Evolutia globala a suprafetei DC, separata per centru/periferie')
plt.xlabel("T (min)")
plt.ylabel("px^2")
print(np.nanmean(avgdiff))
print(np.nanmax(avgdiff))
print(nov6_exp.times[np.nanargmax(avgdiff)])

plt.show()

avgdiff = []    
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            
            if c.colornorm[t] <0.5:
                avgc.append(c.major[t])
                stdc.append(c.major[t])
            else:
                avgp.append(c.major[t])
                
                stdp.append(c.major[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
   
    avgdiff.append((avgp-avgc)/avgc)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
    
plt.title('Evolutia globala a axei majore DC, separata per centru/periferie')
plt.xlabel("T (min)")
plt.ylabel("px")
print(np.nanmean(avgdiff))
print(np.nanmax(avgdiff))
print(nov6_exp.times[np.nanargmax(avgdiff)])

plt.show()

avgdiff = []    
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            
            if c.colornorm[t] <0.5:
                avgc.append(c.ar[t])
                stdc.append(c.ar[t])
            else:
                avgp.append(c.ar[t])
                
                stdp.append(c.ar[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
   
    avgdiff.append((avgp-avgc)/avgc)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
    
plt.title('Evolutia globala a aspect ratio DC separata per centru/periferie')
plt.xlabel("T (min)")
plt.ylabel("Maj/Min")
print(np.nanmean(avgdiff))
print(np.nanmax(avgdiff))
print(nov6_exp.times[np.nanargmax(avgdiff)])
plt.show()
avgdiff = []    
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            
            if c.colornorm[t] <0.5:
                avgc.append(c.interior.area[t])
                stdc.append(c.interior.area[t])
            else:
                avgp.append(c.interior.area[t])
                
                stdp.append(c.interior.area[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
   
    avgdiff.append((avgp-avgc)/avgc)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
    
plt.title('Evolutia globala a suprafetei AC, separata per centru/periferie')
plt.xlabel("T (min)")
plt.ylabel("px^2")
print(np.nanmean(avgdiff))
print(np.nanmax(avgdiff))
print(nov6_exp.times[np.nanargmax(avgdiff)])

plt.show()

avgdiff = []    
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            
            if c.colornorm[t] <0.5:
                avgc.append(c.interior.major[t])
                stdc.append(c.interior.major[t])
            else:
                avgp.append(c.interior.major[t])
                
                stdp.append(c.interior.major[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
   
    avgdiff.append((avgp-avgc)/avgc)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
    
plt.title('Evolutia globala a axei majore AC, separata per centru/periferie')
plt.xlabel("T (min)")
plt.ylabel("px")
print(np.nanmean(avgdiff))
print(np.nanmax(avgdiff))
print(nov6_exp.times[np.nanargmax(avgdiff)])

plt.show()

avgdiff = []    
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            
            if c.colornorm[t] <0.5:
                avgc.append(c.interior.ar[t])
                stdc.append(c.interior.ar[t])
            else:
                avgp.append(c.interior.ar[t])
                
                stdp.append(c.interior.ar[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
   
    avgdiff.append((avgp-avgc)/avgc)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
    
plt.title('Evolutia globala a aspect ratio AC separata per centru/periferie')
plt.xlabel("T (min)")
plt.ylabel("Maj/Min")
print(np.nanmean(avgdiff))
print(np.nanmax(avgdiff))
print(nov6_exp.times[np.nanargmax(avgdiff)])

plt.show()
#%%
data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i:
            sollist = [c.area[t] for t in c.times]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px^2")
plt.title("Probabilitati estimate pentru suprafata DC, per generatii")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2")
plt.xlabel("Generatii")
plt.title("Boxplots pentru suprafata DC, per generatii")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC Area stats")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i:
            sollist = [c.major[t] for t in c.times]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px")
plt.title("Probabilitati estimate pentru axa majora DC, per generatii")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px")
plt.xlabel("Generatii")
plt.title("Boxplots pentru axa majora DC, per generatii")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC Major stats")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i:
            sollist = [c.ar[t] for t in c.times]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("Maj/Min")
plt.title("Probabilitati estimate pentru AR DC, per generatii")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("Maj/Min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru AR DC, per generatii")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC AR stats")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i:
            sollist = [c.interior.area[t] for t in c.times]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px^2")
plt.title("Probabilitati estimate pentru suprafata AC, per generatii")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2")
plt.xlabel("Generatii")
plt.title("Boxplots pentru suprafata AC, per generatii")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC Area stats")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i:
            sollist = [c.interior.major[t] for t in c.times]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px")
plt.title("Probabilitati estimate pentru axa majora AC, per generatii")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px")
plt.xlabel("Generatii")
plt.title("Boxplots pentru axa majora AC, per generatii")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC Major stats")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i:
            sollist = [c.interior.ar[t] for t in c.times]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("Maj/Min")
plt.title("Probabilitati estimate pentru AR AC, per generatii")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("Maj/Min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru AR AC, per generatii")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC AR stats")
print(outlierpoints)
print(medians)
print(stds)

#%% Plot pante per celula toate gen DC
outliers = []
for c in dec6_exp.cells:
    if c.node.generation < 7 and c.node.generation != 0:
        c.makeprovparent()
        c.plotvalue('area',t_shift = 0,addouts = False)
        c.provisionalparent = False
        if c.outlier:
            outliers.append(c)
plt.title("Evolutia suprafetei DC pentru toate celulele")
plt.xlabel("T (min)")
plt.ylabel('Area (px^2)')
plt.show()

for c in dec6_exp.cells:
    if c.node.generation < 7 and c.node.generation != 0:
        c.makeprovparent()
        c.plotvalue('major',t_shift = 0,addouts = False)
        c.provisionalparent = False
plt.title("Evolutia axei majore DC pentru toate celulele")
plt.xlabel("T (min)")
plt.ylabel('Major (px)')
plt.show()
for c in dec6_exp.cells:
    if c.node.generation < 7 and c.node.generation != 0:
        c.makeprovparent()
        c.plotvalue('ar',t_shift = 0,addouts = False)
        c.provisionalparent = False
plt.title("Evolutia aspect ratio DC pentru toate celulele")
plt.xlabel("T (min)")
plt.ylabel('AR (Major/Minor)')
plt.show()
for c in outliers:
    plt.plot(c.times,[c.area[t] for t in c.times])
    print(c.name)
plt.show()

#%% Plot pante per celula gens 1-4 DC
for c in nov6_exp.cells:
    if c.node.generation < 5 and c.node.generation != 0:
        c.makeprovparent()
        c.plotvalue('area',t_shift = 0,addouts = False)
        c.provisionalparent = False
plt.title("Evolutia suprafetei DC pentru generatii 1-4")
plt.xlabel("T (min)")
plt.ylabel('Area (px^2)')
plt.show()

for c in nov6_exp.cells:
    if c.node.generation < 5 and c.node.generation != 0:
        c.makeprovparent()
        c.plotvalue('major',t_shift = 0,addouts = False)
        c.provisionalparent = False
plt.title("Evolutia axei majore DC pentru generatii 1-4")
plt.xlabel("T (min)")
plt.ylabel('Major (px)')
plt.show()
for c in nov6_exp.cells:
    if c.node.generation < 5 and c.node.generation != 0:
        c.makeprovparent()
        c.plotvalue('ar',t_shift = 0,addouts = False)
        c.provisionalparent = False
plt.title("Evolutia aspect ratio DC pentru generatii 1-4")
plt.xlabel("T (min)")
plt.ylabel('AR (Major/Minor)')
plt.show()
#%% Per generatie DC
plotpergeneration2D(nov6_exp,'area')
plt.show()
plotpergeneration2D(nov6_exp,'major')
plt.show()
plotpergeneration2D(nov6_exp,'ar')
plt.show()
#%% Plot celula/periferie gen 1-4
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <4:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.areaslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta suprafetei in DC (px^2/min)')
plt.title("Pantele suprafetei DC vs distanta medie de la centrul de masa")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <4:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.majorslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta axei majore in DC (px/min)')
plt.title("Pantele axei majore DC vs distanta medie de la centrul de masa")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <4:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.arslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta aspect ratio in DC (Major/Minor/min)')
plt.title("Pantele aspect ratio DC vs distanta medie de la centrul de masa")
plt.show()

#%% Plot celula/periferie toate gen
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.areaslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta suprafetei in DC (px^2/min)')
plt.title("Pantele suprafetei DC vs dCOM: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.majorslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta axei majore in DC (px/min)')
plt.title("Pantele axei majore DC vs dCOM: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.arslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta aspect ratio in DC (Major/Minor/min)')
plt.title("Pantele aspect ratio DC vs dCOM: gen 1-6")
plt.show()
#%% Per cc, DC: gen 1-4 
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5 and not c.outlier:
        plt.scatter(c.duration,c.areaslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta suprafetei in DC (px^2/min)')
plt.title("Pantele suprafetei DC vs durata CC: gen 1-4")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5 and not c.outlier:
        plt.scatter(c.duration,c.majorslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta axei majore in DC (px/min)')
plt.title("Pantele axei majore DC vs durata CC: gen 1-4")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5 and not c.outlier:
        plt.scatter(c.duration,c.arslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta aspect ratio in DC (Major/Minor/min)')
plt.title("Pantele aspect ratio DC vs durata CC: gen 1-4")
plt.show()

#%% Per cc, DC: toate gen
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(c.duration,c.areaslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta suprafetei in DC (px^2/min)')
plt.title("Pantele suprafetei DC vs durata CC: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(c.duration,c.majorslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta axei majore in DC (px/min)')
plt.title("Pantele axei majore DC vs durata CC: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(c.duration,c.arslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta aspect ratio in DC (Major/Minor/min)')
plt.title("Pantele aspect ratio DC vs durata CC: gen 1-6")
plt.show()
#%% Statistici primele 4 gen
data = []
lengths = []
medians = []
stds = []
for i in range(1,4):
    gendata = []
    for c in dec6_exp.cells:
        
        if c.node.generation == i  and not c.outlier:
            
            sollist = [c.areaslope]
            gendata += sollist
    lengths.append(len(gendata))
    print(gendata)
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)

plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px^2/min")
plt.title("Probabilitati estimate pentru panta suprafetei DC, gen 1-4")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta suprafetei DC, gen 1-4")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,3):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC Area stats 14")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,4):
    gendata = []
    for c in dec6_exp.cells:
        if c.node.generation == i and not c.outlier:
            sollist = [c.areaslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px/min")
plt.title("Probabilitati estimate pentru panta suprafetei DC, gen 1-4")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta axei majore DC, gen 1-4")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,3):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC major stats 14")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,4):
    gendata = []
    for c in dec6_exp.cells:
        if c.node.generation == i and not c.outlier:
            sollist = [c.areaslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("Major/Minor/min")
plt.title("Probabilitati estimate pentru panta aspect ratio DC, gen 1-4")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta suprafetei DC, gen 1-4")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,3):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC AR stats 14")
print(outlierpoints)
print(medians)
print(stds)
#%% Statistica pe toate generatiile
data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier:
            sollist = [c.areaslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px^2/min")
plt.title("Probabilitati estimate pentru panta suprafetei DC, gen 1-6")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta suprafetei DC, gen 1-6")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC Area stats 16")
print(outlierpoints)
print(medians)
print(stds)
areaslopes = []
majorslopes = []
arslopes = []
areaR2s = []
majorR2s = []
arR2s = []
def slopeandR(x, y):

    slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
    return slope,r_value**2

for i in range(1,7):
    timeslist = []
    arealist = []
    majorlist = []
    arlist = []
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier:
            timeslist += [t-c.times[0] for t in c.times]
            arealist += [c.area[t] for t in c.times]
            majorlist += [c.major[t] for t in c.times]
            arlist += [c.ar[t] for t in c.times]
    # timeslist,arealist,majorlist,arlist = list(zip(*sorted(zip(timeslist,arealist,majorlist,arlist))))
    areaslope,areaR2 = slopeandR(timeslist,arealist)
    majorslope,majorR2 = slopeandR(timeslist,majorlist)
    arslope,arR2 = slopeandR(timeslist,arlist)
    areaslopes.append(areaslope)
    majorslopes.append(majorslope)
    arslopes.append(arslope)
    areaR2s.append(areaR2)
    majorR2s.append(majorR2)
    arR2s.append(arR2)
print("Arealope")
print(areaslopes)
print(areaR2s)
print("major")
print(majorslopes)
print(majorR2s)
print("AR")
print(arR2s)
print(arslopes)
            
data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i  and not c.outlier:
            sollist = [c.majorslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px/min")
plt.title("Probabilitati estimate pentru panta axei majore DC, gen 1-6")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta axei majore DC, gen 1-6")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC major stats 16")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i  and not c.outlier:
            sollist = [c.arslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("Major/Minor/min")
plt.title("Probabilitati estimate pentru panta aspect ratio DC, gen 1-6")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("Major/Minor/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta aspect ratio DC, gen 1-6")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("DC AR stats 16")
print(outlierpoints)
print(medians)
print(stds)
#%% Pantele per celula AC toate celulele
for c in nov6_exp.cells:
    if c.node.generation < 7 and c.node.generation != 0 and not c.outlier:
        c.makeprovparent()
        plt.plot(c.times,[c.interior.area[t] for t  in c.times], c = plt.cm.magma(c.colornorm[c.times[0]]))
        c.provisionalparent = False
plt.title("Evolutia suprafetei AC pentru toate celulele")
plt.xlabel("T (min)")
plt.ylabel('px^2')
plt.show()

for c in nov6_exp.cells:
    if c.node.generation < 7 and c.node.generation != 0 and not c.outlier:
        c.makeprovparent()
        plt.plot(c.times,[c.interior.major[t] for t  in c.times], c = plt.cm.magma(c.colornorm[c.times[0]]))
        c.provisionalparent = False
plt.title("Evolutia axei majore AC pentru toate celulele")
plt.xlabel("T (min)")
plt.ylabel('px^2')
plt.show()
for c in nov6_exp.cells:
    if c.node.generation < 7 and c.node.generation != 0 and not c.outlier:
        c.makeprovparent()
        plt.plot(c.times,[c.interior.ar[t] for t  in c.times], c = plt.cm.magma(c.colornorm[c.times[0]]))
        c.provisionalparent = False
plt.title("Evolutia aspect ratio AC pentru toate celulele")
plt.xlabel("T (min)")
plt.ylabel('px^2')
plt.show()

#%% Pantele per celula AC gen 1- 4
for c in nov6_exp.cells:
    if c.node.generation < 5 and c.node.generation != 0 and not c.outlier:
        c.makeprovparent()
        plt.plot(c.times,[c.interior.area[t] for t  in c.times], c = plt.cm.magma(c.colornorm[c.times[0]]))
        c.provisionalparent = False
plt.title("Evolutia suprafetei AC pentru gen 1-4")
plt.xlabel("T (min)")
plt.ylabel('px^2')
plt.show()

for c in nov6_exp.cells:
    if c.node.generation < 5 and c.node.generation != 0 and not c.outlier:
        c.makeprovparent()
        plt.plot(c.times,[c.interior.major[t] for t  in c.times], c = plt.cm.magma(c.colornorm[c.times[0]]))
        c.provisionalparent = False
plt.title("Evolutia axei majore AC pentru gen 1-4")
plt.xlabel("T (min)")
plt.ylabel('px^2')
plt.show()
for c in nov6_exp.cells:
    if c.node.generation < 5 and c.node.generation != 0 and not c.outlier:
        c.makeprovparent()
        plt.plot(c.times,[c.interior.ar[t] for t  in c.times], c = plt.cm.magma(c.colornorm[c.times[0]]))
        c.provisionalparent = False
plt.title("Evolutia aspect ratio AC pentru gen 1-4")
plt.xlabel("T (min)")
plt.ylabel('px^2')
plt.show()

#%% per generatie AC
for i in range(1,7):
    for c in nov6_exp.cells:
        if c .node.generation == i and not c.outlier:
            plt.plot(c.times,[c.interior.area[t] for t in c.times],c = plt.cm.magma(c.colornorm[c.times[0]]))
            plt.scatter(c.times,[c.interior.area[t] for t in c.times],c = [plt.cm.magma(c.colornorm[t]) for t in c.times])
    plt.xlabel('T (min)')
    plt.ylabel('px^2')
    plt.title('Generatia '+str(i)+': suprafata')
    plt.show()
    
for i in range(1,7):
    for c in nov6_exp.cells:
        if c .node.generation == i and not c.outlier:
            plt.plot(c.times,[c.interior.major[t] for t in c.times],c = plt.cm.magma(c.colornorm[c.times[0]]))
            plt.scatter(c.times,[c.interior.major[t] for t in c.times],c = [plt.cm.magma(c.colornorm[t]) for t in c.times])
    plt.xlabel('T (min)')
    plt.ylabel('px')
    plt.title('Generatia '+str(i)+': axa majora')
    plt.show()
    
for i in range(1,7):
    for c in nov6_exp.cells:
        if c .node.generation == i and not c.outlier:
            plt.plot(c.times,[c.interior.ar[t] for t in c.times],c = plt.cm.magma(c.colornorm[c.times[0]]))
            plt.scatter(c.times,[c.interior.ar[t] for t in c.times],c = [plt.cm.magma(c.colornorm[t]) for t in c.times])
    plt.xlabel('T (min)')
    plt.ylabel('Major/Minor')
    plt.title('Generatia '+str(i)+': AR')
    plt.show()
#%% Plot celula/periferie  AC gen 1-4
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation < 6 and not c.outlier and c.interior.areaslope >0:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.interior.areaslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta suprafetei in AC (px^2/min)')
plt.title("Pantele suprafetei AC vs distanta medie de la centrul de masa")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5 and not c.outlier and c.interior.majorslope >0:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.interior.majorslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta axei majore in AC (px/min)')
plt.title("Pantele axei majore AC vs distanta medie de la centrul de masa")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5 and not c.outlier and c.interior.arslope >0:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.interior.arslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta aspect ratio in AC (Major/Minor/min)')
plt.title("Pantele aspect ratio AC vs distanta medie de la centrul de masa")
plt.show()

#%% Plot celula/periferie  ACtoate gen
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7  and not c.outlier and c.interior.areaslope >0:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.interior.areaslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta suprafetei in AC (px^2/min)')
plt.title("Pantele suprafetei AC vs dCOM: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7  and not c.outlier and c.interior.majorslope > 0:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.interior.majorslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta axei majore in AC (px/min)')
plt.title("Pantele axei majore AC vs dCOM: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7  and not c.outlier and c. interior.arslope >0:
        plt.scatter(np.mean([c.density[t] for t in c.times]),c.interior.arslope,c = 'r')
plt.xlabel('Distanta de la COM  medie (px)')
plt.ylabel('Panta aspect ratio in AC (Major/Minor/min)')
plt.title("Pantele aspect ratio AC vs dCOM: gen 1-6")
plt.show()
#%% Per cc, AC: gen 1-4 
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5  and not c.outlier:
        plt.scatter(c.duration,c.areaslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta suprafetei in AC (px^2/min)')
plt.title("Pantele suprafetei AC vs durata CC: gen 1-4")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5 and not c.outlier:
        plt.scatter(c.duration,c.majorslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta axei majore in AC (px/min)')
plt.title("Pantele axei majore AC vs durata CC: gen 1-4")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <5 and not c.outlier:
        plt.scatter(c.duration,c.arslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta aspect ratio in AC (Major/Minor/min)')
plt.title("Pantele aspect ratio AC vs durata CC: gen 1-4")
plt.show()

#%% Per cc, AC: toate gen
for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(c.duration,c.areaslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta suprafetei in AC (px^2/min)')
plt.title("Pantele suprafetei AC vs durata CC: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(c.duration,c.majorslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta axei majore in AC (px/min)')
plt.title("Pantele axei majore AC vs durata CC: gen 1-6")
plt.show()

for c in nov6_exp.cells:
    if c.node.generation >0 and c.node.generation <7 and not c.outlier:
        plt.scatter(c.duration,c.arslope,c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.xlabel('Durata (min)')
plt.ylabel('Panta aspect ratio in AC (Major/Minor/min)')
plt.title("Pantele aspect ratio AC vs durata CC: gen 1-6")
plt.show()

#%% Statistici AC gen 1-4
data = []
lengths = []
medians = []
stds = []
for i in range(1,5):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i  and not c.outlier and c.interior.areaslope >0:
            sollist = [c.interior.areaslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px^2/min")
plt.title("Probabilitati estimate pentru panta suprafetei AC, gen 1-4")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta suprafetei AC, gen 1-4")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,4):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC Area stats 14")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,5):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier and c.interior.majorslope >0:
            sollist = [c.interior.majorslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px/min")
plt.title("Probabilitati estimate pentru panta suprafetei AC, gen 1-4")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta axei majore AC, gen 1-4")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,4):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC major stats 14")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,5):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier and c.interior.arslope >0:
            sollist = [c.interior.arslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("Major/Minor/min")
plt.title("Probabilitati estimate pentru panta aspect ratio AC, gen 1-4")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta aspect ratio AC, gen 1-4")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,4):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC AR stats 14")
print(outlierpoints)
print(medians)
print(stds)
#%% Statistici AC toate gen
data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i  and not c.outlier and c.interior.areaslope >0:
            sollist = [c.interior.areaslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px^2/min")
plt.title("Probabilitati estimate pentru panta suprafetei AC, gen 1-6")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px^2/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta suprafetei AC, gen 1-6")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC Area stats 16")
print(outlierpoints)
print(medians)
print(stds)
areaslopes = []
majorslopes = []
arslopes = []
areaR2s = []
majorR2s = []
arR2s = []
for i in range(1,7):
    timeslist = []
    arealist = []
    majorlist = []
    arlist = []
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier:
            timeslist += [t-c.times[0] for t in c.times]
            arealist += [c.interior.area[t] for t in c.times]
            majorlist += [c.interior.major[t] for t in c.times]
            arlist += [c.interior.ar[t] for t in c.times]
    # timeslist,arealist,majorlist,arlist = list(zip(*sorted(zip(timeslist,arealist,majorlist,arlist))))
    areaslope,areaR2 = slopeandR(timeslist,arealist)
    majorslope,majorR2 = slopeandR(timeslist,majorlist)
    arslope,arR2 = slopeandR(timeslist,arlist)
    areaslopes.append(areaslope)
    majorslopes.append(majorslope)
    arslopes.append(arslope)
    areaR2s.append(areaR2)
    majorR2s.append(majorR2)
    arR2s.append(arR2)
print("Arealope")
print(areaslopes)
print(areaR2s)
print("major")
print(majorslopes)
print(majorR2s)
print("AR")
print(arR2s)
print(arslopes)
data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier and c.interior.majorslope >0:
            sollist = [c.interior.majorslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("px/min")
plt.title("Probabilitati estimate pentru panta axei majore AC, gen 1-6")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("px/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta axei majore AC, gen 1-6")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC major stats 16")
print(outlierpoints)
print(medians)
print(stds)

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i and not c.outlier and c.interior.arslope >0:
            sollist = [c.interior.arslope]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)
 
plt.violinplot(data,showmedians = True)
plt.xlabel("Generatii")
plt.ylabel("Major/Minor/min")
plt.title("Probabilitati estimate pentru panta aspect ratio AC, gen 1-6")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("pMajor/MInor/min")
plt.xlabel("Generatii")
plt.title("Boxplots pentru panta aspect ratio AC, gen 1-6")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print("AC AR stats 16")
print(outlierpoints)
print(medians)
print(stds)
#%%
dec6_exp.plotaverage('cellcount',color = 'k')
nov6_exp.plotaverage('cellcount',color = 'r')
plt.show()

totalavg = []
totalstd = []
for t in dec6_exp.times:
    avg = []
    std = []
    for c in dec6_exp.cells:
        if t in c.times:
            value = c.ACsolidity[t]
            avg.append(value)
            std.append(value)
    totalavg.append(np.mean(avg))
    totalstd.append(np.std(std))

plt.errorbar([t+20 for t in dec6_exp.times],totalavg,yerr = totalstd,c = 'k')

totalavg = []
totalstd = []
for t in nov12_exp.times:
    avg = []
    std = []
    for c in nov12_exp.cells:
        if t in c.times:
            value = c.ACsolidity[t]
            avg.append(value)
            std.append(value)
    totalavg.append(np.mean(avg))
    totalstd.append(np.std(std))

plt.errorbar([t+20 for t in nov12_exp.times],totalavg,yerr = totalstd,c = 'purple')
        
totalavg = []
totalstd = []
for t in nov6_exp.times:
    avg = []
    std = []
    for c in nov6_exp.cells:
        if t in c.times:
            value = c.ACsolidity[t]
            avg.append(value)
            std.append(value)
    totalavg.append(np.mean(avg))
    totalstd.append(np.std(std))

plt.errorbar(nov6_exp.times,totalavg,yerr = totalstd,c = 'r')

totalavg = []
totalstd = []
for t in nov5_exp.times:
    avg = []
    std = []
    for c in nov5_exp.cells:
        if t in c.times:
            value = c.ACsolidity[t]
            avg.append(value)
            std.append(value)
    totalavg.append(np.mean(avg))
    totalstd.append(np.std(std))

plt.errorbar([t+20 for t in nov5_exp.times],totalavg,yerr = totalstd,c = 'g')


plt.show()
#%%
nov12_exp.plotaverage('mean',color = 'r')
dec6_exp.plotaverage('mean')
plt.show()
    
#%%
for c in nov6_exp.cells:
    plt.plot([t-c.times[0] for t in c.times],[c.area[t] for t in c.times],c = plt.cm.magma(c.colornorm[c.times[0]]))
plt.title('Evolutia soliditatii per celula, suprapuse')
plt.xlabel("T (min)")
plt.ylabel("Suprafata AC/ Suprafata DC")

plt.show()   

for i in range(1,7):
    for c in nov6_exp.cells:
        if c.node.generation == i:
            c.makeprovparent()
            c.plotvalue('area')
            c.provisionalparent = False
    plt.title('Evolutia soliditatii per generatii: gen = '+str(i))
    plt.xlabel("T (min)")
    plt.ylabel("Suprafata AC/ Suprafata DC")
    plt.show()

avgdiff = []    
for t in nov6_exp.times:
     
    avgc = []
    stdc = []
    avgp = []
    stdp = []
    for c in nov6_exp.cells:
        if t in c.times:
            
            if c.colornorm[t] <0.5:
                avgc.append(c.area[t])
                stdc.append(c.area[t])
            else:
                avgp.append(c.area[t])
                
                stdp.append(c.area[t])
    avgc = np.mean(avgc)
    avgp = np.mean(avgp)
    stdc = np.std(stdc)
    stdp = np.std(stdp)
   
    avgdiff.append((avgp-avgc)/avgc)
    plt.errorbar(t,avgc,yerr = stdc,c = 'purple',fmt = 'o')
    plt.errorbar(t,avgp,yerr = stdp,c = 'orange',fmt = 'o')
    
plt.title('Evolutia globala a soliditatii, separata per centru/periferie')
plt.xlabel("T (min)")
plt.ylabel("Suprafata AC/ Suprafata DC")
print(np.nanmean(avgdiff))
print(np.nanmax(avgdiff))
print(nov6_exp.times[np.nanargmax(avgdiff)])

plt.show()

for c in nov6_exp.cells:
    if c.duration >=10:
        plt.errorbar(c.duration,np.mean([c.area[t] for t in c.times]),yerr = np.std([c.area[t] for t in c.times]),c = plt.cm.magma(c.times[0]),fmt = 'o')
    
plt.title('SARia medie per celula vs durata estimata ciclului celular')
plt.xlabel("Durata (min)")
plt.ylabel("px^2")
plt.show()

data = []
lengths = []
medians = []
stds = []
for i in range(1,7):
    gendata = []
    for c in nov6_exp.cells:
        if c.node.generation == i:
            sollist = [c.area[t] for t in c.times]
            gendata += sollist
    lengths.append(len(gendata))
    medians.append(np.median(gendata))
    stds.append(np.std(gendata))
    data.append(gendata)

plt.violinplot(data,showmedians = True)

plt.ylabel("Suprafata AC/Suprafata DC")
plt.xlabel("Generatii")
plt.title("Probabilitati estimate pentru valoarea soliditatii, per generatie")
plt.legend()

plt.show()
res = plt.boxplot(data)
plt.ylabel("Suprafata AC/Suprafata DC")
plt.xlabel("Generatii")
plt.title("Boxplots pentru valoarea soliditatii, per generatie")
plt.legend()

plt.show()
outlierpoints = []
for i in range(0,6):
    fliers = res['fliers'][i].get_ydata()
    outlierpoints.append(len(fliers)/lengths[i])
print(outlierpoints)
print(medians)
print(stds)
#for t in nov12_exp.times:
#     avg = []
#     std = []
#     for c in nov12_exp.cells:
#         if t in c.times:
#             value = c.interior.area[t]/c.area[t]
#             avg.append(value)
#             std.append(value)
#     avg = np.mean(avg)
#     std = np.std(std)
#     plt.errorbar(t+20,avg,yerr = std,c = 'purple',fmt = 'o')


# for t in nov5_exp.times:
#     avg = []
#     std = []
#     for c in nov5_exp.cells:
#         if t in c.times:
#             value = c.interior.area[t]/c.area[t]
#             avg.append(value)
#             std.append(value)
#     avg = np.mean(avg)
#     std = np.std(std)
#     plt.errorbar(t+40,avg,yerr = std,c = 'g',fmt = 'o')
# plt.show()
def plotlineage(node,value_name,splitlist,col = 0.0,single = False,tshift = 0):
    
    if col != 0.0:
        node.cell.color = col
    node.cell.plotvalue(value_name,t_shift = tshift,addouts = True)
    plt.scatter(node.cell.times[0],node.cell.__dict__[value_name][node.cell.times[0]],c = node.cell.color)
    plt.scatter(node.cell.times[-1],node.cell.__dict__[value_name][node.cell.times[-1]],c = node.cell.color)
    print('Plotted cell '+str(node.cell.name)+" of generation "+str(node.generation))
    #plt.axvline(x=node.cell.times[0])
    if node.children != []:
        if not single or node.generation in splitlist:
            for c in node.children:
                plotlineage(c,value_name,splitlist,col = col,single = single,tshift = tshift)
        else:
                
            plotlineage(node.children[0],value_name,splitlist,col = col,single = single,tshift = tshift )
            
# for r in nov6_exp.lineagetrees:
    
#     for c in r.children:
#         for c2 in c.children:
#             r.cell.plotvalue('solidity')
#             c.cell.plotvalue('solidity')
#             plotlineage(c2)
#             plt.ylim(0.2,0.75)
#             plt.ylabel("Solidity")
#             plt.xlabel("T")
#             plt.show()


def plotpergeneration3D(experiment):
    roots = experiment.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
    for i in range(1,mx-1):
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d", elev=30, azim=130)
        for c in experiment.cells:
            if c.node.generation == i and not c.outlier:
                if c.duration > 15 and c.duration < 60:
                    c.makeprovparent()
                    xvalues, yvalues, duration,colorlist = c.plotvalue('solidity',plotflag=False)
                    c.provisionalparent = False
                    ax.plot(xvalues,[c.duration for x in xvalues],yvalues,c = plt.cm.magma(colorlist[0,0]))
                    ax.scatter(xvalues,[c.duration for x in xvalues],yvalues,c = colorlist,cmap = 'magma',norm = 'linear')
                    
        ax.set_xlabel("T")
        ax.set_ylabel("Duration")
        ax.set_ylim(0,90)
        ax.set_zlabel("Solidity")
        plt.title("Generatia "+str(i))
        plt.show()
        
    for i in range(1,mx-1):
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d", elev=30, azim=40)
        for c in experiment.cells:
            if c.node.generation == i and not c.outlier:
                if c.duration > 15 and c.duration < 60:
                    c.makeprovparent()
                    xvalues, yvalues, duration,colorlist = c.plotvalue('solidity',plotflag=False)
                    c.provisionalparent = False
                    ax.plot(xvalues,[c.duration for x in xvalues],yvalues,c = plt.cm.magma(colorlist[0,0]))
                    ax.scatter(xvalues,[c.duration for x in xvalues],yvalues,c = colorlist,cmap = 'magma',norm = 'linear')
                
        ax.set_xlabel("T")
        ax.set_ylabel("Duration")
        ax.set_ylim(0,60)
        ax.set_zlabel("Solidity")
        plt.title("Generatia "+str(i))
        plt.show()
        
def plotgenduration(experiment):
    roots = experiment.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
    for i in range(1,mx-1):
        for c in experiment.cells:
            if c.node.generation == i and not c.outlier:
                if c.duration > 10:
                    plt.scatter(i,c.duration,c = plt.cm.magma(c.colornorm[c.times[0]]))
                
    plt.xlabel("Generatie")      
    plt.ylabel("Durata (min)")      
    plt.ylim(0,65)
    plt.show()
    
def plotpergeneration2D(experiment,value_name,plot_outliers=False):
    roots = experiment.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
    for i in range(1,mx):
        
        for c in experiment.cells:
            if c.node.generation == i and not c.outlier:
                if c.duration > 0:
                    c.makeprovparent()
                    time, value, duration,colors = c.plotvalue(value_name,plotflag=False)
                    c.provisionalparent = False
                    plt.scatter(time,value,c = colors)
                    plt.plot(time ,value,c = colors[0])
                
        plt.xlabel("T")
        plt.ylabel(value_name)
        plt.title("Generatia "+str(i))
        plt.show()
    if plot_outliers:
        for i in range(1,mx):
            
            for c in experiment.cells:
                if c.node.generation == i and c.outlier:
                    if c.duration > 0:
                        print(i)
                        print(c.name)
                        print(c.parent.name)
                        c.makeprovparent()
                        time, value, duration,colors = c.plotvalue(value_name,plotflag=False)
                        c.provisionalparent = False
                        plt.scatter(time,value,c = colors)
                        plt.plot(time ,value,c = colors[0])
                    
            plt.xlabel("T")
            plt.ylabel(value_name)
            plt.title("Outliers Generatia "+str(i))
            plt.show()
        
def avgpergeneration2D(experiment,value_name  ='solidity',batch = False,intervalbin = 1.5,cycletime = 18,antibiotic = None,norm = True,scaling = True):
    roots = experiment.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
    generationsavg = []
    generationstimes = []
    generationsstd = []
    for i in range(mx):
        genvalues= {}
        gentimes = []
        
        for c in experiment.cells:
            if c.node.generation == i and not c.outlier:
                if c.duration > 0:
                    if antibiotic == None or c.times[-1] < antibiotic:
                        c.makeprovparent()
                        time, value, duration,colors = c.plotvalue(value_name,plotflag= False)
                        time = [x-time[0] for x in time]
                        d = time[-1]-time[0]
                        if scaling:
                            if d == 0:
                                time = [cycletime/2]
                            else:
                                scaling = cycletime/duration[0]
                                time = [x*scaling for x in time]
                            if norm:
                                value = [(x - np.mean(value))/np.std(value) for x in value]
                        else:
                            if d > cycletime:
                                cycletime = d
                            m,b = np.polyfit(x, y, 1)
        
                        times = {t:v for t ,v in zip(time,value)}
                        c.provisionalparent = False
                        
                        for t in times.keys():
                            if t not in gentimes:
                                gentimes.append(t)
                                genvalues.update({t:[times[t]]})
                            else:
                                if type(genvalues[t])!= list:
                                    genvalues[t] = [genvalues[t]]
                                genvalues[t].append(times[t])
        
        
        gentimes = list(sorted(gentimes))
        intervals = [x*intervalbin for x in range(int(cycletime//intervalbin+2))]
        binnedtimes = intervals[:-1].copy()
        binnedavg = intervals[:-1].copy()
        binnedstd = intervals[:-1].copy()
        for k in range(len(intervals)-1):
            temp = []
            for key in genvalues.keys():
                if intervals[k] <= key and intervals[k+1] >= key:
                    temp += genvalues[key]
            if len(temp) >3:
                binnedavg[k] = np.mean(temp)
                binnedstd[k] = np.std(temp)
            else:
                binnedavg[k] = None
                binnedstd[k] = None
        # avg = [np.mean(genvalues[t]) for t in gentimes]
        # std = [np.std(genvalues[t]) for t in gentimes]
        binnedtimes = [t for t,x in zip(binnedtimes,binnedavg) if x != None]
        
        binnedavg = [x for x in binnedavg if x!=None]
        binnedstd = [x for x in binnedstd if x!=None]
        generationsavg.append(binnedavg)
        generationstimes.append(binnedtimes)
        generationsstd.append(binnedstd)
        
        if not batch:
            plt.errorbar(binnedtimes ,binnedavg,yerr = binnedstd)
        # plt.ylim(0.2,0.6)
        plt.xlabel("T")
        plt.ylabel(value_name)
        plt.title("Generatia "+str(i))
        if not batch:
            plt.show()
    if batch:
        return generationstimes,generationsavg,generationsstd
        
def rollavgpergeneration(experiment,value_name  ='solidity',batch = False,window_size = 3,antibiotic = None,norm = True,scaling = True):
    roots = experiment.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
    generationsavg = []
    generationstimes = []
    generationsstd = []
    for i in range(mx):
        genvalues= {}
        gentimes = []
        
        for c in experiment.cells:
            if c.node.generation == i and not c.outlier:
                if c.duration > 0:
                    if antibiotic == None or c.times[-1] < antibiotic:
                        c.makeprovparent()
                        time, value, duration,colors = c.plotvalue(value_name,plotflag= False)
                        time = [x-time[0] for x in time]
                        d = time[-1]-time[0]
        
                        times = {t:v for t ,v in zip(time,value)}
                        c.provisionalparent = False
                        
                        for t in times.keys():
                            if t not in gentimes:
                                gentimes.append(t)
                                genvalues.update({t:[times[t]]})
                            else:
                                if type(genvalues[t])!= list:
                                    genvalues[t] = [genvalues[t]]
                                genvalues[t].append(times[t])
        gentimes = list(sorted(gentimes))
        valuelist = []
        avglist = []
        for t in gentimes:
            valuelist.append(genvalues[t])
        for i in range(len(valuelist)-window_size+1):
            out =[]
            w = valuelist[i:i+window_size]
            for el in w:
                out += el
            out = np.mean(out)
            avglist.append(out)
        generationsavg.append(avglist)
        generationstimes.append(gentimes)
    print(generationsavg)
    return generationstimes,generationsavg

def getgenslopes(experiment,value_name  ='solidity',antibiotic = None):
    roots = experiment.lineagetrees
    mx = 0
    for r in roots:
        depth = r.maxdepthdown()
        if depth>mx:
            mx = depth
    genslopes = []
    gens = []
    for i in range(1,mx):
        slopes = []
        
        for c in experiment.cells:
            if c.node.generation == i and not c.outlier:
                if c.duration > 0:
                    if antibiotic == None or c.times[-1] < antibiotic:
                        c.makeprovparent()
                        time, value, duration,colors = c.plotvalue(value_name,plotflag= False)
                        time = [x-time[0] for x in time]
                        if len(time) != 1:
                            m,b = np.polyfit(time,value,1)
                            slopes.append(m)
        genslopes.append(np.mean(slopes))
        gens.append(i)
    return genslopes,gens                         
        
        
nov6_exp.getcellcount()
nov6_exp.plotaverage('cellcount',color = 'red')
nov12_exp.getcellcount()
nov12_exp.plotaverage('cellcount',color = 'purple')
plt.show()
#%%
nov6_exp.getcellcount()
nov6_exp.plotaverage('cellcount',color = 'red')
plt.xlabel("T(min)")
plt.ylabel("# de celule")
plt.show()
plotgenduration(nov12_exp)

plt.show()
#%%
nov12_exp.getcellcount()
nov12_exp.plotaverage('cellcount',color = 'purple')
plt.axvline(x = 70,c = 'k')
plt.xlabel("T(min)")
plt.ylabel("# de celule")
plt.show()
plotgenduration(nov12_exp)
plt.show()
#%%
nov6_exp.getcellcount()
nov6_exp.plotaverage('cellcount',color = 'red')
nov12_exp.getcellcount()
count = [nov12_exp.cellcount[t] for t in nov12_exp.times]
times = [t+20 for t in nov12_exp.times]
plt.scatter(times,count,c = 'purple')
plt.axvline(x = 90,c = 'k')
plt.xlabel("T(min)")
plt.ylabel("# de celule")
plt.show()
#%%
time = [t for t in nov6_exp.times if t <125]
y = [nov6_exp.cellcount[x] for x in time]
def exp(t,tau):
    return 2**(t/tau)
exponential = Model(exp)
result = exponential.fit(y,t = time, tau = 20)
print(result.fit_report())

# time = nov42corr_exp.times
# y = [nov42corr_exp.cellcount[x] for x in time]
# def exp(t,tau):
#     return 2**(t/tau)
# exponential = Model(exp)
# result = exponential.fit(y,t = time, tau = 20)
# print(result.fit_report())

# time = sept6_exp.times
# y = [sept6_exp.cellcount[x] for x in time]
# def exp(t,tau):
#     return 2**(t/tau)
# exponential = Model(exp)
# result = exponential.fit(y,t = time, tau = 20)
# print(result.fit_report())
def exp(t,tau):
    return 2**(t/tau)+2
time = [t for t in nov12_exp.times if t >100]
y = [nov12_exp.cellcount[x] for x in time]
def exp(t,tau):
    return 2**(t/tau)
exponential = Model(exp)
result = exponential.fit(y,t = time, tau = 20)
print(result.fit_report())

# plotgenduration(nov12_exp)
# plotgenduration(sept6_exp)
# plotpergeneration3D(nov6_exp)
# intervals = [[20,45],[45,60]]
# for i in intervals:
#     clist = [x.duration for x in nov6_exp.cells if x.duration > i[0] and x.node.generation == 6 and x.duration <i[1]]
#     avg = np.mean(clist)
#     std = np.std(clist)
#     plt.errorbar(6,avg,yerr = std,fmt = 'o')
#     count = len(clist)
#     print("#Gen 6 between "+str(i)+": "+str(count))
    

# for i in range(1,6):
#     clist = [x.duration for x in nov6_exp.cells if x.node.generation == i and x.duration < 60]
#     avg = np.mean(clist)
#     std = np.std(clist)
#     plt.errorbar(i,avg,yerr = std,fmt = 'o')
#     count = len(clist)
#     print("#Gen "+str(i)+" : "+str(count))
# plt.show()


#t42,a42,std42 = avgpergeneration2D(nov42corr_exp,'solidity',batch = True,norm = False)
t6,a6,std6 = avgpergeneration2D(nov6_exp,'solidity',batch = True,norm = False)
#tsept,asept,stdsept = avgpergeneration2D(sept6_exp,'solidity',batch = True,norm = False)
t12,a12,std12 = avgpergeneration2D(nov12_exp,'solidity',batch = True,norm = False)

# def offsetnestedlist(l):
#     for i in range(len(l)):
#         if l[i] == []:
#             continue
#         temp = np.array(l[i])
#         temp = (temp -np.mean(temp))/np.std(temp)
#         l[i] = list(temp)

def outputslope(t,v,exp,gen):
    if len(t) == 0:
        return None
    y = v[-1]-v[0]
    x = t[-1]-t[0]
    s = y/x
    txt = str(exp)+" - generatia "+str(gen)+": "+str(s)
    print(txt)
    return s
# # for i in [a42,a6,asept,a12]:
# #     offsetnestedlist(i)

for i in range(len(t6)):
    plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
    
    if i <len(t12)+1 and i >0 and i<7:
        plt.errorbar(t12[i-1],a12[i-1],std12[i-1],c = 'purple',fmt = '--')
    plt.title("Generatia "+str(i)+" solidity")
    plt.show()

t = np.linspace(0,18,50)
sin = 0.1*np.sin(2*np.pi*0.1*t)+0.5

for i in range(1,len(t6)):
    plt.plot(t6[i],a6[i],c = 'red')
    plt.scatter(t6[i],a6[i],c = 'red')
    if i <len(t12)+1 and i >0 and i<8:
        plt.plot(t12[i-1],a12[i-1],c = 'purple')
        plt.scatter(t12[i-1],a12[i-1],c = 'purple')

    plt.title("Generatia "+str(i)+" solidity")
    plt.show()

for i in range(1,len(t6)-1):
    markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
    [bar.set_alpha(0.3) for bar in bars]
    if i <len(t12)+1 and i >0 and i < 6:
        markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-1]],a12[i-1],yerr = std12[i-1],c = 'purple')
        [bar.set_alpha(0.3) for bar in bars]
    plt.axvline(i*18)
plt.title("Grafic soliditate pentru generatiile normalizate")
plt.figure(figsize = (16,6))
plt.show()
times6 = []
values6 = []
times12 = []
values12 =[]

for i in range(1,len(t6)):
    times6 += [t+i*18 for t in t6[i]]
    values6 += a6[i]
   
    if i <len(t12)+1 and i >0 and i < 6:
        times12 += [t+i*18 for t in t12[i-1]]
        values12 += a12[i-1]
        
    plt.axvline(i*18)
plt.plot(times6,values6,c = 'red')
plt.plot(times12,values12,c = 'purple')
plt.title("Evolutia generica a soliditatii")
plt.figure(figsize = (16,6))
plt.show()
#%%
value = 'major'
vavg = 'major_avg'
exp = nov12_exp
exp.plotaverage(value,color = 'purple')
m,b = np.polyfit(exp.times,exp.__dict__[vavg],1)
print(m,b)
plt.xlabel("T(min)")
plt.ylabel("Major axis (px)")
plt.show()


#%%  


t42,a42,std42 = avgpergeneration2D(nov42corr_exp,'major',batch = True,norm = False)
t6,a6,std6 = avgpergeneration2D(nov6_exp,'major',batch = True,norm = False)
tsept,asept,stdsept = avgpergeneration2D(sept6_exp,'major',batch = True,norm = False)
t12,a12,std12 = avgpergeneration2D(nov12_exp,'major',batch = True,norm = False)



for i in range(1,len(t6)):
    plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
    if i <len(t12)+1 and i >0 and i<6:
        plt.errorbar(t12[i-1],a12[i-1],std12[i-1],c = 'purple',fmt = '--')
    plt.title("Generatia "+str(i)+" major")
    plt.show()

t = np.linspace(0,18,50)
sin = 0.1*np.sin(2*np.pi*0.1*t)+0.5

print("\nEllMaj slopes\n")
s6 = [None for x in t6]
s6t = s6.copy()
s42 = [None for x in t42]
s42t = s42.copy()
ssept = [None for x in tsept]
sseptt = ssept.copy()
s12 = [None for x in t12]
s12t = s12.copy()

for i in range(1,len(t6)):
    plt.plot(t6[i],a6[i],c = 'red')
    plt.scatter(t6[i],a6[i],c = 'red')
    s6[i] = outputslope(t6[i],a6[i],'nov 6 ctrl',i)
    s6t[i] = i
    
    
    if i <len(t12)+1 and i >1 and i<6:
        plt.plot(t12[i-1],a12[i-1],c = 'purple')
        plt.scatter(t12[i-1],a12[i-1],c = 'purple')
        s12[i] = outputslope(t12[i-1],a12[i-1],'nov 12 antibiotic',i)
        s12t[i] = i
    print('\n')
    # if i == 3:
    #     plt.plot(t,sin,c = 'k')
    plt.title("Generatia "+str(i)+' major')
    plt.show()
    
s6,s6t = getgenslopes(nov6_exp,'major')
s12,s12t = getgenslopes(nov12_exp,'major')
print(s6)
plt.scatter(s6t,s6,c = 'red')
plt.plot(s6t,s6,c = 'red')
plt.plot(s42t,s42,c = 'green')
plt.plot(sseptt,ssept,c = 'blue')
plt.plot(s12t,s12,c = 'purple')
plt.scatter(s12t,s12,c = 'purple')
plt.title("Major axis slope per generation")
plt.xlabel("Gen")
plt.ylabel("Slope")
plt.show()

for i in range(1,len(t6)-1):
   markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
   [bar.set_alpha(0.3) for bar in bars]
   if i <len(t12)+1 and i >0 and i < 6:
       markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-1]],a12[i-1],yerr = std12[i-1],c = 'purple')
       [bar.set_alpha(0.3) for bar in bars]
   plt.axvline(i*18)
plt.title("Grafic axa major pentru generatiile normalizate")
plt.figure(figsize = (16,6))
plt.show()


#%%%
t42,a42,std42 = avgpergeneration2D(nov42corr_exp,'area',batch = True,norm = False)
t6,a6,std6 = avgpergeneration2D(nov6_exp,'area',batch = True,norm = False,scaling=False,intervalbin=3)
tsept,asept,stdsept = avgpergeneration2D(sept6_exp,'area',batch = True,norm = False)
t12,a12,std12 = avgpergeneration2D(nov12_exp,'area',batch = True,norm = False,scaling=False,intervalbin=3)



for i in range(1,len(t6)):
    plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
    if i <len(t12)+1 and i >0 and i<6:
        plt.errorbar(t12[i-1],a12[i-1],std12[i-1],c = 'purple',fmt = '--')
    plt.title("Generatia "+str(i) +" area")
    plt.show()

t = np.linspace(0,18,50)
sin = 0.1*np.sin(2*np.pi*0.1*t)+0.5

print("\nArea slopes\n")
s6 = [None for x in t6]
s6t = s6.copy()
s42 = [None for x in t42]
s42t = s42.copy()
ssept = [None for x in tsept]
sseptt = ssept.copy()
s12 = [None for x in t12]
s12t = s12.copy()

for i in range(1,len(t6)):
    plt.plot(t6[i],a6[i],c = 'red')
    plt.scatter(t6[i],a6[i],c = 'red')
    s6[i] = outputslope(t6[i],a6[i],'nov 6 ctrl',i)
    s6t[i] = i
    if i <len(t12)+1 and i >0 and i<6:
        plt.plot(t12[i-1],a12[i-1],c = 'purple')
        plt.scatter(t12[i-1],a12[i-1],c = 'purple')
        s12[i] = outputslope(t12[i-1],a12[i-1],'nov 12 antibiotic',i)
        s12t[i] = i
    print('\n')
    # if i == 3:
    #     plt.plot(t,sin,c = 'k')
    plt.title("Generatia "+str(i)+" area")
    plt.show()
plt.plot(s6t,s6,c = 'red')
plt.scatter(s6t,s6,c = 'red')
plt.plot(s42t,s42,c = 'green')
plt.plot(sseptt,ssept,c = 'blue')
plt.plot(s12t,s12,c = 'purple')
plt.scatter(s12t,s12,c = 'purple')
plt.title("Area slope per generation")
plt.xlabel("Gen")
plt.ylabel("Slope")
plt.show()

for i in range(len(t6)-1):
    markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
    [bar.set_alpha(0.3) for bar in bars]
    if i <len(t12)+1 and i >0 and i < 6:
        markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-1]],a12[i-1],yerr = std12[i-1],c = 'purple')
        [bar.set_alpha(0.3) for bar in bars]
    plt.axvline(i*18)
plt.title("Grafic area  pentru generatiile normalizate")
plt.figure(figsize = (16,6))
plt.show()

#%%
t42,a42,std42 = avgpergeneration2D(nov42corr_exp,'ar',batch = True,norm = False)
t6,a6,std6 = avgpergeneration2D(nov6_exp,'ar',batch = True,norm = False)
tsept,asept,stdsept = avgpergeneration2D(sept6_exp,'ar',batch = True,norm = False)
t12,a12,std12 = avgpergeneration2D(nov12_exp,'ar',batch = True,norm = False)



for i in range(1,len(t6)):
    plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
    if i <len(t12)+1 and i >0 and i<6:
        plt.errorbar(t12[i-1],a12[i-1],std12[i-1],c = 'purple',fmt = '--')
    plt.title("Generatia "+str(i) + " aspect ratio")
    plt.show()

t = np.linspace(0,18,50)
sin = 0.1*np.sin(2*np.pi*0.1*t)+0.5

print("\nAR slopes\n")
s6 = [None for x in t6]
s6t = s6.copy()
s42 = [None for x in t42]
s42t = s42.copy()
ssept = [None for x in tsept]
sseptt = ssept.copy()
s12 = [None for x in t12]
s12t = s12.copy()

for i in range(1,len(t6)):
    plt.plot(t6[i],a6[i],c = 'red')
    plt.scatter(t6[i],a6[i],c = 'red')
    s6[i] = outputslope(t6[i],a6[i],'nov 6 ctrl',i)
    s6t[i] = i
    if i <len(t12)+1 and i >0 and i<6:
        plt.plot(t12[i-1],a12[i-1],c = 'purple')
        plt.scatter(t12[i-1],a12[i-1],c = 'purple')
        s12[i] = outputslope(t12[i-1],a12[i-1],'nov 12 antibiotic',i)
        s12t[i] = i
    print('\n')
    # if i == 3:
    #     plt.plot(t,sin,c = 'k')
    plt.title("Generatia "+str(i)+ " aspect ratio")
    plt.show()

        

plt.plot(s6t,s6,c = 'red')
plt.scatter(s6t,s6,c = 'red')
plt.plot(s42t,s42,c = 'green')
plt.plot(sseptt,ssept,c = 'blue')
plt.plot(s12t,s12,c = 'purple')
plt.scatter(s12t,s12,c = 'purple')
s6t = s6t[1:3]
s6 = s6[1:3]
s12t = s12t[1:3]
s12 = s12[1:3]
avg= [(x+y)/2 for x, y in zip(s6,s12)]
plt.plot(s12t,avg,c = 'k')
plt.scatter(s12t,avg,c = 'k')
plt.title("ARslope per generation")
plt.xlabel("Gen")
plt.ylabel("Slope")
plt.show()

for i in range(len(t6)-1):
   markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
   [bar.set_alpha(0.3) for bar in bars]
   if i <len(t12)+1 and i >0 and i < 6:
       markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-1]],a12[i-1],yerr = std12[i-1],c = 'purple')
       [bar.set_alpha(0.3) for bar in bars]
   plt.axvline(i*18)
plt.title("Grafic aspect ratio pentru generatiile normalizate")
plt.figure(figsize = (16,6))
plt.show()

#%%

t42,a42,std42 = avgpergeneration2D(nov42corr_exp,'skew',batch = True,norm = False)
t6,a6,std6 = avgpergeneration2D(nov6_exp,'skew',batch = True,norm = False)
tsept,asept,stdsept = avgpergeneration2D(sept6_exp,'skew',batch = True,norm = False)
t12,a12,std12 = avgpergeneration2D(nov12_exp,'skew',batch = True,norm = False)




for i in range(1,len(t6)-1):
    plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
    if i <len(t12)+1 and i >0 and i<6:
        plt.errorbar(t12[i-1],a12[i-1],std12[i-1],c = 'purple',fmt = '--')
    plt.title("Generatia "+str(i) + " skew")
    plt.show()

t = np.linspace(0,18,50)
sin = 0.1*np.sin(2*np.pi*0.1*t)+0.5
for i in range(1,len(t6)):
    plt.plot(t6[i],a6[i],c = 'red')
    plt.scatter(t6[i],a6[i],c = 'red')
    if i <len(t12)+1 and i >0 and i<6:
        plt.plot(t12[i-1],a12[i-1],c = 'purple')
        plt.scatter(t12[i-1],a12[i-1],c = 'purple')
    # if i == 3:
    #     plt.plot(t,sin,c = 'k')
    plt.title("Generatia "+str(i) + " skew")
    plt.show()

for i in range(len(t6)-1):
    markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
    [bar.set_alpha(0.3) for bar in bars]
    if i <len(t12)+1 and i >0 and i < 6:
        markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-1]],a12[i-1],yerr = std12[i-1],c = 'purple')
        [bar.set_alpha(0.3) for bar in bars]
    plt.axvline(i*18)
plt.title("Grafic skew pentru generatiile normalizate")
plt.figure(figsize = (16,6))
plt.show()


#%%
t42,a42,std42 = avgpergeneration2D(nov42corr_exp,'kurt',batch = True,norm = False)
t6,a6,std6 = avgpergeneration2D(nov6_exp,'kurt',batch = True,norm = False)
tsept,asept,stdsept = avgpergeneration2D(sept6_exp,'kurt',batch = True,norm = False)
t12,a12,std12 = avgpergeneration2D(nov12_exp,'kurt',batch = True,norm = False)





for i in range(1,len(t6)-1):
    plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
    if i <len(t12)+1 and i >0 and i<6:
        plt.errorbar(t12[i-1],a12[i-1],std12[i-1],c = 'purple',fmt = '--')
    plt.title("Generatia "+str(i)+ " kurt")
    plt.show()

t = np.linspace(0,18,50)
sin = 0.1*np.sin(2*np.pi*0.1*t)+0.5
for i in range(1,len(t6)-3):
    plt.plot(t6[i],a6[i],c = 'red')
    plt.scatter(t6[i],a6[i],c = 'red')
    if i <len(t12)+1 and i >0 and i<5:
        plt.plot(t12[i-1],a12[i-1],c = 'purple')
        plt.scatter(t12[i-1],a12[i-1],c = 'purple')
    # if i == 3:
    #     plt.plot(t,sin,c = 'k')
    plt.title("Generatia "+str(i)+ " kurt")
    plt.show()

for i in range(len(t6)-1):
    markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
    [bar.set_alpha(0.3) for bar in bars]
    if i <len(t12)+1 and i >0 and i < 6:
        markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-1]],a12[i-1],yerr = std12[i-1],c = 'purple')
        [bar.set_alpha(0.3) for bar in bars]
    plt.axvline(i*18)
plt.title("Grafic kurt pentru generatiile normalizate")
plt.figure(figsize = (16,6))
plt.show()
#%%










#%%

t42,a42,std42 = avgpergeneration2D(nov42corr_exp,'mean',batch = True,norm = False)
t6,a6,std6 = avgpergeneration2D(nov6_exp,'mean',batch = True,norm = False)
tsept,asept,stdsept = avgpergeneration2D(sept6_exp,'mean',batch = True,norm = False)
t12,a12,std12 = avgpergeneration2D(nov12_exp,'mean',batch = True,norm = False)





for i in range(1,len(t6)):
    plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
    if i <len(t12)+1 and i >0 and i<6:
        plt.errorbar(t12[i-1],a12[i-1],std12[i-1],c = 'purple',fmt = '--')
    plt.title("Generatia "+str(i)+ " mean")
    plt.show()

t = np.linspace(0,18,50)
sin = 0.1*np.sin(2*np.pi*0.1*t)+0.5
for i in range(1,len(t6)-1):
    plt.plot(t6[i],a6[i],c = 'red')
    plt.scatter(t6[i],a6[i],c = 'red')
    if i <len(t12)+1 and i >0 and i<6:
        plt.plot(t12[i-1],a12[i-1],c = 'purple')
        plt.scatter(t12[i-1],a12[i-1],c = 'purple')
    # if i == 3:
    #     plt.plot(t,sin,c = 'k')
    plt.title("Generatia "+str(i)+ " mean")
    plt.show()
    
# plotpergeneration2D(nov12_exp,'solidity')
# plotpergeneration2D(nov6_exp,'solidity')
# avgpergeneration2D(nov6_exp,'area')


# plotpergeneration2D(nov6_exp)

# nov6_exp.plotcells()

#%%
def plotparsinglelineage(exp,valuelist,splits):
    for v in valuelist:
        for n in exp.lineagetrees:
            plotlineage(n,v,splitlist = splits,single=True)
            plt.title(str(v)+" vs time: "+str(exp.name))
            plt.xlabel("T(min)")
            plt.ylabel(v)
            plt.show()
            
def plotmultiplelineages(exp_list,value_list,splits, color_list =None,startindex = None,tlimits = None,tshift = [0],gen = [0]):
    for v in value_list:
        print(exp_list)
        for exp in exp_list:
            
            eindex = exp_list.index(exp)
            children = [c.node for c in exp.cells if c.node.generation == gen[eindex]]
            print([c.cell.name for c in children])
            if type(color_list) == list:
                
                col = color_list[eindex]
            else:
                col = 0.0
            if startindex == None:
                n = children[0]
            else: 
                index = startindex[eindex]
                n = children[index]
            print(n.cell.name)
            n.cell.makeprovparent()
            plotlineage(n,v,splitlist = splits, col = col,single = True,tshift=tshift[eindex])
            exp.getdensity()
            exp.normalizecellattr('color')
            n.cell.provisionalparent = False
        title = "+".join([e.name for e in exp_list])
        title += ": "+v
        plt.title(title)
        plt.xlabel("T(min")
        
       # plt.xlim(exp_list[0].times[2],exp_list[0].times[-1])
        plt.ylabel(v)
        plt.show()
def rollingaverage(l,window = 3):
    length = len(l)
    out = []
    for i in range(length-window+1):
        w = l[i:i+window]
        avg = sum(w)/window
        out.append(avg)
    return out

def plotmultiplelineages(exp_list,value_list,splits, color_list =None,startindex = None,tlimits = None,tshift = [0],gen = [0]):
    for v in value_list:
        print(exp_list)
        for exp in exp_list:
            
            eindex = exp_list.index(exp)
            children = [c.node for c in exp.cells if c.node.generation == gen[eindex]]
            print([c.cell.name for c in children])
            if type(color_list) == list:
                
                col = color_list[eindex]
            else:
                col = 0.0
            if startindex == None:
                n = children[0]
            else: 
                index = startindex[eindex]
                n = children[index]
            print(n.cell.name)
            n.cell.makeprovparent()
            plotlineage(n,v,splitlist = splits, col = col,single = True,tshift=tshift[eindex])
            exp.getdensity()
            exp.normalizecellattr('color')
            n.cell.provisionalparent = False
        title = "+".join([e.name for e in exp_list])
        title += ": "+v
        plt.title(title)
        plt.xlabel("T(min")
        
       # plt.xlim(exp_list[0].times[2],exp_list[0].times[-1])
        plt.ylabel(v)
        plt.show()
#%%    
plotpergeneration2D(nov6_exp,'ACsolidity')
#%%
valuelist = ['mean','solidity','skew','kurt','major','ar','area']
value = ['mean']

splits = [1,2]
si = [[1,2,1]]

explist = [dec6_exp]
clist = ['r','purple']
tshifts = [0,0,0]
gens = [0,0,0]
# plotparsinglelineage(nov12_exp,valuelist)
nov12_exp.name = '12 nov 10mgL'
# for s in si:
#     plotmultiplelineages(explist, value, splits, clist,startindex = s,tshift = tshifts,gen = gens)
plotparsinglelineage(nov6_exp,value,splits)


for c in nov12_exp.cells:
    if c.name == 'Track_109.b':
        z = c.node
# nov6_exp.plotcells()




#%%

nov12_exp.antibiotic = 0
nov6_exp.plotaverage('massratio',color = 'r')
nov12_exp.plotaverage('massratio',color = 'purple')
plt.xlabel('T(min)')
plt.ylabel('Avg cell mean')
plt.show()


#
#%%

value = 'ACsolidity'
t6,a6,std6 = avgpergeneration2D(dec6_exp,value,batch = True,norm =False,intervalbin = 1.5)
#t5,a5,std5 = avgpergeneration2D(nov5_exp,value,batch = True,norm = False,intervalbin = 1.5)
t12,a12,std12 = avgpergeneration2D(nov12_exp,value,batch = True,norm = False,intervalbin = 1.5)
# for t,a in zip(t12,a12):
#     if not np.any(t):
#         continue
#     m,b,c = np.polyfit(t,a,2)
#     print(m,b,c)
# for i in range(1,9):
#     if i < len(t6)-1:
#         plt.errorbar(t6[i],a6[i],yerr = std6[i],c = 'red',fmt = '--')
#     if i <len(t12)+2 and i >1:
#         plt.errorbar(t12[i-2],a12[i-2],std12[i-2],c = 'purple',fmt = '--')
#     plt.title("Generatia "+str(i)+ " "+value)
#     plt.show()


# for i in range(1,9):
#     if i < len(t6):
#         plt.plot(t6[i],a6[i],c = 'red')
#         plt.scatter(t6[i],a6[i],c = 'red')
#     if i <len(t12)+2 and i > 1:
#         plt.plot(t12[i-2],a12[i-2],c = 'purple')
#         plt.scatter(t12[i-2],a12[i-2],c = 'purple')
#     plt.title("Generatia "+str(i)+ " "+value)
#     plt.show()



# for i in range(1,len(t6)-1):
#     markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
#     [bar.set_alpha(0.3) for bar in bars]
#     if i <len(t12)+2and i < len(t6)-1:
#         markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-2]],a12[i-2],yerr = std12[i-2],c = 'purple')
#         [bar.set_alpha(0.3) for bar in bars]
#     plt.axvline(i*18)
# plt.title("Grafic "+value+" pentru generatiile normalizate")
# plt.figure(figsize = (16,6))
# plt.show()
plt.figure(figsize = (16,6))
for i in range(1,9):
    if i <len(t6):
        markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
        [bar.set_alpha(0.3) for bar in bars]
    if i < len(t12)+2 and i >1:
        markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-2]],a12[i-2],yerr = std12[i-2],c = 'purple')
        [bar.set_alpha(0.3) for bar in bars]
    # if i < len(t5)+3 and i >2:
    #     markers,caps, bars = plt.errorbar([t+i*18 for t in t5[i-3]],a5[i-3],yerr = std5[i-3],c = 'g')
    #     [bar.set_alpha(0.3) for bar in bars]
    # # plt.axvline(i*18)


plt.title("Grafic "+value+" pentru generatiile normalizate")
plt.show()

s6,s6t = getgenslopes(nov6_exp,value)
s12,s12t = getgenslopes(nov12_exp,value)
s12t =  [t+1 for t in s12t[:-1]]
s12 = [x for x in s12[:-1]]
plt.plot(s6t[:-1],s6[:-1],c = 'r',alpha = 0.3)
plt.plot(s12t,s12,c = 'purple',alpha = 0.3)
plt.scatter(s6t[:-1],s6[:-1],c = 'r',alpha = 0.9)
plt.scatter(s12t,s12,c = 'purple',alpha = 0.9)
plt.ylabel(value)
plt.xlabel("Generatia")
plt.title("Evolutia pantei pentru parametrul "+str(value))
plt.show() 

#%%
value = 'mean'
t6,a6,std6 = avgpergeneration2D(nov6_exp,value,batch = True,norm = False,intervalbin = 3)
#tsept,asept,stdsept = avgpergeneration2D(sept6_exp,value,batch = True,norm = False)
#t12,a12,std12 = avgpergeneration2D(nov12_exp,value,batch = True,norm = False,intervalbin = 3)

for t,a in zip(t12,a12):
    if not np.any(t):
        continue
    m,b,c = np.polyfit(t,a,2)
    print(m,b,c)
c = 'red'
# for i in range(1,9):
#     if i < len(t6)-1:
#         plt.errorbar(t6[i],a6[i],yerr = std6[i],c = c,fmt = '--')
#     plt.title("Generatia "+str(i)+ " "+value)
#     plt.show()


# for i in range(1,9):
#     if i < len(t6):
#         plt.plot(t6[i],a6[i],c = c)
#         plt.scatter(t6[i],a6[i],c = c)
#     plt.title("Generatia "+str(i)+ " "+value)
#     plt.show()



# for i in range(1,len(t6)-1):
#     markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = 'red',label = '6 noiembrie')
#     [bar.set_alpha(0.3) for bar in bars]
#     if i <len(t12)+2and i < len(t6)-1:
#         markers,caps, bars = plt.errorbar([t+i*18 for t in t12[i-2]],a12[i-2],yerr = std12[i-2],c = 'purple')
#         [bar.set_alpha(0.3) for bar in bars]
#     plt.axvline(i*18)
# plt.title("Grafic "+value+" pentru generatiile normalizate")
# plt.figure(figsize = (16,6))
# plt.show()
plt.figure(figsize = (16,6))
for i in range(1,5):
    if i <len(t6):
        markers,caps, bars =plt.errorbar([t+i*18 for t in t6[i]],a6[i],yerr = std6[i],c = c,label = '6 noiembrie')
        [bar.set_alpha(0.3) for bar in bars]
    # plt.axvline(i*18)

plt.title("Grafic "+value+" pentru generatiile normalizate")
plt.show()

s6,s6t = getgenslopes(nov6_exp,value)


plt.plot(s6t[:-1],s6[:-1],c = c,alpha = 0.3)

plt.scatter(s6t[:-1],s6[:-1],c = c,alpha = 0.9)

plt.ylabel(value)
plt.xlabel("Generatia")
plt.title("Evolutia pantei pentru parametrul "+str(value))
plt.show() 

#%%
for i in range(1,6):
    for c in dec6_exp.cells:
        if c.node.generation == i:
            plt.plot(c.times,[c.interior.hmean[t]/c.hmean[t] for t in c.times],c = 'r')
    # for c in nov5_exp.cells:
    #     if c.node.generation == i-2:
    #         plt.plot(c.times,[c.interior.hmean[t]/c.hmean[t] for t in c.times],c = 'g')
    for c in nov12_exp.cells:
        if c.node.generation == i-1:
            plt.plot(c.times,[c.interior.hmean[t]/c.hmean[t] for t in c.times],c = 'purple')
    plt.show()

 

