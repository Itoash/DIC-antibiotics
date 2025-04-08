# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
import cv2
import concurrent.futures
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

cimport numpy as np
from libc.math cimport sqrt
from libc.float cimport FLT_MAX

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t DTYPE_int_t

def obtain_network(list imgs, double scorethreshold=0.5):
    # imgs: list of 2D numpy arrays as images
    # scorethreshold: double to set in imagenetwork class
    
    cdef list processedimgs = imgs.copy()
    processedimgs = remaplabels(processedimgs)
    
    cdef tuple result = multiprocessframes(processedimgs)
    cdef list stackdata = result[0]
    cdef list dist = result[1]
    
    cdef list tracklist = []
    stackdata.append(stackdata[-1])
    
    cdef int i
    for i in range(len(dist)):
        tracklist.append(evaluateframe(processedimgs[i], stackdata[i], stackdata[i+1], dist[i], i))
    
    stackdata = stackdata[:-1]
    cdef object network = imagenetwork(processedimgs, tracklist, stackdata, scorethreshold=scorethreshold)
    return network
 
def remaplabels(list stack):
    cdef int i
    cdef np.ndarray img
    cdef int label
    
    for i, img in enumerate(stack):
        if len(np.unique(img)) != np.max(stack[i])+1:
            for i, label in enumerate(np.unique(img)):
                img[img==label] = i         

    return stack

def multiprocessframes(list imagestack):
    cdef list opstack = imagestack.copy()
    opstack.append(np.zeros_like(opstack[0]))  # zero-pad stack
    
    cdef list stackdata
    with concurrent.futures.ThreadPoolExecutor() as executor:
        stackdata = list(executor.map(process_frame, opstack[:-1], opstack[1:], 
                                      np.arange(0, len(imagestack))))
    
    cdef list COMlist = []
    cdef list COMs
    cdef np.ndarray COMs_array
    
    for el in stackdata:
        COMs = [i for i in el[0]]
        COMs_array = np.asarray(COMs)
        COMlist.append(COMs_array)
    
    COMlist.append(COMlist[-1])
    
    cdef list dist
    with concurrent.futures.ThreadPoolExecutor() as executor:
        dist = list(executor.map(inversedistancematrix, COMlist[:-1], COMlist[1:]))
    
    return stackdata, dist

def process_frame(np.ndarray currentimage, np.ndarray nextimage, int imageid):
    cdef dict overlapdict = {}
    cdef dict COMdict = {}
    cdef dict angledict = {}
    cdef dict majordict = {}
    cdef dict minordict = {}
    cdef dict areadict = {}
    cdef dict convexdict = {}
    
    cdef list COMlist = []
    cdef list arealist = []
    cdef list angleslist = []
    cdef list overlaplist = []
    cdef list arlist = []
    cdef list majorlist = []
    
    # Ensure proper label indexing
    if len(np.unique(currentimage)) != np.max(currentimage)+1:
        for i, label in enumerate(np.unique(currentimage)):
            currentimage[currentimage==label] = i
            
    if len(np.unique(nextimage)) != np.max(nextimage)+1:
        for i, label in enumerate(np.unique(nextimage)):
            nextimage[nextimage==label] = i
    
    cdef int col
    cdef tuple dims
    cdef np.ndarray filtered
    cdef np.ndarray coords
    cdef np.ndarray com
    cdef double area
    cdef double angle
    cdef double major
    cdef double minor
    cdef np.ndarray overlaps
    cdef tuple counts_tuple
    cdef np.ndarray counts
    cdef tuple contour_data
    cdef np.ndarray contour
    cdef tuple ellipse
    cdef tuple cdims
    
    for col in range(1, int(np.max(currentimage))+1):
        dims = np.shape(currentimage)
        
        filtered = currentimage.copy()
        filtered.flatten('F')
        filtered[filtered != float(col)] = 0
        filtered = filtered/col
        filtered = filtered.astype(np.float64)
        filtered = np.reshape(filtered, dims)
        
        # coordinates for all points
        coords = np.argwhere(filtered==1)
        # centroid coords
        com = np.mean(coords, axis=0)
        
        area = np.sum(filtered)
        
        if np.sum(filtered) > 1:
            contour_data = cv2.findContours(filtered.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = contour_data[0][0]
            
            cdims = np.shape(contour)
            contour = np.reshape(contour, (cdims[0], cdims[2]))
            
            if len(contour[:,0]) > 4:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                major = ellipse[1][1]
                minor = ellipse[1][0]
            else:
                angle = 0
                major = 0
                minor = 0
                
            overlaps = np.multiply(filtered, nextimage)
            counts_tuple = np.unique(overlaps, return_counts=True)
            counts = np.asarray(counts_tuple)
            
            if len(counts[0,:]) > 1:
                counts = counts[:, 1:]
                counts[1,:] /= np.sum(filtered)
            else:
                counts = None
        else:
            angle = 0
            major = 1
            minor = 1
            counts = 1
            
        overlapdict.update({col: counts})
        COMdict.update({col: com})
        angledict.update({col: angle})
        majordict.update({col: major})
        minordict.update({col: minor})
        areadict.update({col: area})
        
        COMlist.append(com)
        arealist.append(area)
        angleslist.append(angle)
        overlaplist.append(counts)
        
        if minor != 0:
            arlist.append(np.nan_to_num(major/minor))
        else:
            arlist.append(0)
            
        majorlist.append(major)
   
    return (np.asarray(COMlist), np.asarray(arealist), np.asarray(angleslist), 
            np.asarray(arlist), overlaplist, np.asarray(majorlist))

def distancematrix(np.ndarray COMarray1, np.ndarray COMarray2):
    cdef np.ndarray dists = distance_matrix(COMarray1, COMarray2)
    return dists

def inversedistancematrix(np.ndarray COMarray1, np.ndarray COMarray2):
    cdef np.ndarray dists = distance_matrix(COMarray1, COMarray2)
    dists = 1/dists
    dists[dists > 1] = 1
    return dists

def ratiomatrix(np.ndarray array1, np.ndarray array2):
    cdef np.ndarray matrix = np.empty((len(array1), len(array2)))
    
    cdef int i
    for i in range(len(array1)):
        matrix[i,:] = array2/array1[i]

    matrix = matrix.flatten()
    
    cdef np.ndarray largerthanone = np.argwhere(matrix > 1)
    for i in largerthanone:
        matrix[i] = 1/matrix[i]
        
    matrix = np.reshape(matrix, (len(array1), len(array2)))
    
    return matrix  # ratio of every element in array1 (first dim) to every element in array2(second dim)

def evaluateframe(np.ndarray frame, tuple framedict1, tuple framedict2, np.ndarray distancematrix, int ind):
    cdef list distancelist = [distancematrix[i,:] for i in range(distancematrix.shape[0])]
    cdef list angleslist3 = [i for i in framedict1[2]]
    cdef list angleslist4 = [i for i in framedict2[2]]
    cdef list arealist3 = [i for i in framedict1[1]]
    cdef list arealist4 = [i for i in framedict2[1]]
    cdef list arlist3 = [i for i in framedict1[3]]
    cdef list arlist4 = [i for i in framedict2[3]]
    cdef list overlaplist = [i for i in framedict1[4]]
    
    cdef np.ndarray anglematrix = ratiomatrix(np.asarray(angleslist3), np.asarray(angleslist4))
    cdef list angle_list = [anglematrix[i,:] for i in range(anglematrix.shape[0])]
    
    cdef np.ndarray areamatrix = ratiomatrix(np.asarray(arealist3), np.asarray(arealist4))
    cdef list area_list = [areamatrix[i,:] for i in range(areamatrix.shape[0])]
    
    cdef np.ndarray armatrix = ratiomatrix(np.asarray(arlist3), np.asarray(arlist4))
    cdef list ar_list = [armatrix[i,:] for i in range(armatrix.shape[0])]
    
    cdef list frametrack = []
    
    with concurrent.futures.ThreadPoolExecutor() as exe:
        frametrack = list(exe.map(evaluatecandidates, distancelist, overlaplist, 
                                  angle_list, area_list, ar_list, 
                                  np.arange(1, np.max(frame)+1), 
                                  [ind for i in range(len(distancelist))]))
        
    return frametrack

cdef class Node:
    cdef public:
        int name
        int frame
        dict likelychildrenweights
        bint tbd
        list position
        double area
        double angle
        np.ndarray overlaps
        double major
        double minor
        list likelychildren
        list vcoords
        dict metadata
        list likelyparent
        np.ndarray backwardsoverlaps
    
    def __init__(self, int colorname, int frame, list childrenlist, list weightlist):
        self.name = colorname
        self.frame = frame
        self.likelychildrenweights = {}
        self.likelychildrenweights = {c[0]: w[0] for c, w in zip(childrenlist, weightlist)}
        self.tbd = False
        self.position = [0,0]
        self.area = 0.0
        self.angle = 0.0
        self.overlaps = np.array([0])
        self.major = 0.0
        self.minor = 0.0
        self.likelychildren = []
        self.vcoords = []
        self.metadata = {'colname': self.name, 'frame': self.frame}
        self.likelyparent = []
        self.backwardsoverlaps = None
    
    def addchild(self, object child, double weight):
        if child not in self.children.keys():
            self.children.update({child: weight})
    
    def printlineage(self, bint plot=True):
        for c, w in zip(self.children.keys(), self.children.items()):
            plt.plot([self.frame, self.frame+1], [self.name, c], c='k', ms=w[1])
            plt.scatter([self.frame, self.frame+1], [self.name, c], c='k', s=w[1])
    
    def __str__(self):
        return "Label: " + str(self.name) + "; Frame: " + str(self.frame) + "; TBD = " + str(self.tbd)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.name == other.name and self.frame == other.frame
    
    def removeNode(self, Node node, bint debug=False):
        cdef int name = node.name
        cdef np.ndarray coords
        # check if node is in nextframe
        if node.frame == self.frame + 1:
            if debug:
                print("Target node is in next frame. Printing current data...")
                print(f'Name: {self.name},Frame: {self.frame};\n Overlapdict: {self.likelychildrenweights}\n Overlaps: {self.overlaps} \n Backoverlaps: {self.backwardsoverlaps}')
            
            # if in list, remove
            if node in self.likelychildren:
                self.likelychildren.remove(node)
                
            # if in dict, remove and modify existing key values
            if name in self.likelychildrenweights.keys():
                self.likelychildrenweights.pop(name)
                
            copy = deepcopy(self.likelychildrenweights)
            for k in self.likelychildrenweights.keys():
                if k > node.name:
                    copy[k-1] = copy[k]
                    del copy[k]
            self.likelychildrenweights = copy
            
            # Get coords of name, and check that it is not empty, and overlaps is not None
            if self.overlaps is not None:
                coords = np.argwhere(self.overlaps == name)
                if len(coords) != 0:  # always 2D array, so need second dimension
                    self.overlaps = np.delete(self.overlaps, (coords[0,1]), axis=1)
                    
                coords = np.argwhere(self.overlaps[0,:] > name)
                if len(coords) != 0:
                    # always 2D array, so need second dimension
                    self.overlaps[0, coords] -= 1
                    
            if debug:
                print("Target node removed. Printing data...")
                print(f'Name: {self.name},Frame: {self.frame};\n Overlapdict: {self.likelychildrenweights}\n Overlaps: {self.overlaps} \n Backoverlaps: {self.backwardsoverlaps}')
                
        if node.frame == self.frame - 1:
            if debug:
                print("Target node is in previous frame. Printing current data...")
                print(f'Name: {self.name},Frame: {self.frame};\n Overlapdict: {self.likelychildrenweights}\n Overlaps: {self.overlaps} \n Backoverlaps: {self.backwardsoverlaps}')
            
            if self.backwardsoverlaps is not None:
                coords = np.argwhere(self.backwardsoverlaps == name)  # always 2D array
                if len(coords) != 0:
                    self.backwardsoverlaps = np.delete(self.backwardsoverlaps, (coords[0,1]), axis=1)
                
                coords = np.argwhere(self.backwardsoverlaps[0,:] > name)  # always 2D array
                if len(coords) != 0:
                    self.backwardsoverlaps[0, coords] -= 1
                    
            if debug:
                print("Target node removed. Printing data...")
                print(f'Name: {self.name},Frame: {self.frame};\n Overlapdict: {self.likelychildrenweights}\n Overlaps: {self.overlaps} \n Backoverlaps: {self.backwardsoverlaps}')    
                
        if node.frame == self.frame and self.name > node.name:
            print(f"Target is in same frame ({self.name},{self.frame})")
            self.name -= 1

cdef class imagenetwork:
    cdef public:
        list images
        list nodes
        double similarity_threshold
    
    def __init__(self, list images, list nodes, list stackdata, double scorethreshold=0.3):
        self.images = images.copy()
        
        for nds, datas in zip(nodes, stackdata):
            nds = list(sorted(nds, key=lambda x: x.name))
            for i, node in enumerate(nds):
                node.position = datas[0][i]
                node.area = datas[1][i]
                node.angle = datas[2][i]
                node.overlaps = datas[4][i]
                node.major = datas[5][i]
                node.minor = datas[5][i]/datas[3][i]
                
        cdef list copy = []
        for nlist in nodes:
            copy += nlist
            
        self.nodes = copy
        self.similarity_threshold = 0.3
        self.mt_getbackoverlaps()
        self.mt_findchildren()
        self.makeTree()
        self.virtualizeCoords()
    
    def mt_findchildren(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.findchildren, self.nodes)
    
    def mt_getbackoverlaps(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.getbackoverlaps, np.arange(1, len(self.images)))
    
    def findchildren(self, Node n):
        cdef list children = [el[0] for el in n.likelychildrenweights.items()]
        cdef list weights = [el[1] for el in n.likelychildrenweights.items()]
        cdef list childrenlist = [el for el, w in zip(children, weights) if w > self.similarity_threshold]
        
        n.likelychildren = [el for el in self.nodes if el.name in childrenlist and el.frame == n.frame+1]
        
        for node in n.likelychildren:
            node.likelyparent.append(n)
    
    def makeTree(self):
        cdef Node node
        cdef int best_parent_index
        cdef int best_parent_name
        cdef list badparents
        
        for node in self.nodes:
            if len(node.likelyparent) > 1:
                # choose parent with highest overlap
                best_parent_index = np.argmax(node.backwardsoverlaps[1,:])
                best_parent_name = node.backwardsoverlaps[0, best_parent_index]
                badparents = [n for n in node.likelyparent if n.name != best_parent_name]
                
                for p in badparents:
                    p.likelychildren.remove(node)
                    
                node.likelyparent = [n for n in node.likelyparent if n.name == best_parent_name]
    
    def debug(self):
        cdef int i
        for i in range(len(self.images)):
            print(np.unique(self.images[i]))
            print([n.name for n in self.nodes if n.frame == i])
    
    def getbackoverlaps(self, int index):
        cdef int col
        cdef list node
        cdef Node node_obj
        cdef tuple dims
        cdef np.ndarray filtered
        cdef np.ndarray overlaps
        cdef tuple counts_tuple
        cdef np.ndarray counts
        
        for col in range(1, int(np.max(self.images[index])+1)):
            node = [n for n in self.nodes if n.name == col and n.frame == float(index)]
            try:
                node_obj = node[0]
            except IndexError:
                print(node, index, col)
                continue
                
            dims = np.shape(self.images[index])
            
            filtered = self.images[index].copy()
            filtered.flatten('F')
            filtered[filtered != float(col)] = 0
            filtered = filtered/col
            filtered = filtered.astype(np.float64)
            filtered = np.reshape(filtered, dims)
            
            overlaps = np.multiply(filtered, self.images[index-1])
            counts_tuple = np.unique(overlaps, return_counts=True)
            counts = np.asarray(counts_tuple)
            
            if len(counts[0,:]) > 1:
                counts = counts[:, 1:]
                counts[1,:] /= np.sum(filtered)
            else:
                counts = None
                
            node_obj.backwardsoverlaps = counts
    
    def deletenode(self, int name, int frame):
        cdef np.ndarray chosenimage = self.images[frame]
        cdef Node chosennode = [n for n in self.nodes if n.name == name and n.frame == frame][0]
        
        chosenimage[chosenimage == name] = 0
        chosenimage[chosenimage > name] -= 1
        chosennode.tbd = True
        
        for n in self.nodes:
            n.removeNode(chosennode)
            
        self.nodes.remove(chosennode)
        del chosennode
    
    def unlinknode(self, tuple nodepos):
        cdef Node node = [n for n in self.nodes if n.name == nodepos[1] and n.frame == nodepos[0]][0]
        
        for n in self.nodes:
            n.removeNode(node)
            
        node.likelychildren = []
        node.likelyparent = []
        self.makeTree() 
        self.virtualizeCoords()
    
    def main(self):
        return self
    
    def exportGraph(self):
        cdef np.ndarray nodes_pos = np.zeros((len(self.nodes), 2))
        cdef list edge_indexes = []
        cdef dict indexmap = {}
        cdef list texts = []
        cdef list meta = []
        
        # fill up array with nodes
        cdef int i
        cdef Node n
        cdef str t
        cdef int nodeindex
        cdef int chindex
        
        for i, n in enumerate(self.nodes):
            nodes_pos[i, 1] = n.vcoords[1]
            nodes_pos[i, 0] = n.vcoords[0]
            indexmap.update({n: i})
            t = 'L:' + str(n.name) + ';\nT:' + str(n.frame) + ';'
            texts.append(t)
            meta.append((n.frame, n.name))
            
        for n in self.nodes:
            nodeindex = indexmap[n]
            for ch in n.likelychildren:
                chindex = indexmap[ch]
                edge_indexes.append((nodeindex, chindex))
        
        edge_indexes = np.asarray(edge_indexes)
        return nodes_pos, edge_indexes, texts, meta
    
    def traverseTree(self, Node node, list occupiedlist):
        cdef double lowest_y
        
        if len(node.vcoords) == 0:
            lowest_y = np.max(occupiedlist[node.frame])
            node.vcoords = [node.frame, lowest_y+1]
            occupiedlist[node.frame].append(lowest_y+1)
            
        for ch in node.likelychildren:
            self.traverseTree(ch, occupiedlist)
    
    def virtualizeCoords(self):
        cdef Node n
        cdef list occupiedlist
        
        for n in self.nodes:
            n.vcoords = []
            
        occupiedlist = [[0] for i in range(len(self.images))]
        
        for n in self.nodes:
            self.traverseTree(n, occupiedlist)
            
    def linknodes(self, tuple parentcoords, tuple childcoords):
        cdef Node parent = [n for n in self.nodes if n.name == parentcoords[1] and n.frame == parentcoords[0]][0]
        cdef Node child = [n for n in self.nodes if n.name == childcoords[1] and n.frame == childcoords[0]][0]
        
        child.likelyparent = [parent]
        
        if child not in parent.likelychildren:
            parent.likelychildren.append(child)

def evaluatecandidates(distancetoothers,overlaps,anglestoothers,areatoothers,arstoothers,name,frame,minoverlap = 0.15,th = 0,weights = [1,1,1,1,1]):
    # distancetoothers: 1D array in label order
    # overlaps: 2D array with labels in frst dim and overlaps in second dim
    # anglestoothers: 1D array with angle ratio
    # areatoothers: 1D array with area ratio
    
    if isinstance(overlaps,np.ndarray): # if it has substantial overlaps in next frame, continue scoring
        indexes = np.argwhere(overlaps[1,:]>minoverlap)
        if len(indexes) == 0: # no significant overlap
            return Node(name,frame,[],[])
        candidates = overlaps[0,indexes].astype(int)-1
        candidateoverlaps = overlaps[1,indexes]*weights[1]
        
        # double overlap score, then set larger than one values to max to account for half overlaps
        candidatehalfoverlaps = candidateoverlaps*2
        candidatehalfoverlaps[candidatehalfoverlaps > 1] = 1
        
    else: # otherwise return None
   
        return Node(name,frame,[],[])
    # assign no-division scores
    candidateangles = anglestoothers[candidates]*weights[2]
    
    
    
    candidateareas = areatoothers[candidates]*weights[3]
    candidatears = arstoothers[candidates]*weights[4]
    candidatedistances = distancetoothers[candidates]*weights[0]
    candidatedistances[candidatedistances>1] = 1
    
    # add half-area score for good candidates
    candidatehalfareas = candidateareas*2
    candidatehalfareas[candidatehalfareas>1] = 1
    # add half aspect ratio score
    candidatehalfars = candidatears*2
    candidatehalfars[candidatehalfars>1] = 1
    # add up scores; max is 6, anything above a 3 is pretty good with image stabilization on
    candidatescores = (candidateoverlaps+candidateangles+candidateareas+candidatedistances+candidatehalfoverlaps+candidatehalfareas+candidatears+candidatehalfars)/(np.sum(weights)+weights[3]+weights[4]+weights[1])
    candidates +=1
    finalcandidatescores = candidatescores.tolist()
    finalcandidates = candidates.tolist()
    
    _,sortedcandidates =  list(zip(*sorted(zip(finalcandidatescores, finalcandidates))))
    sortedscores = list(sorted(finalcandidatescores))
    # tup = tuple(zip(*[
    # (c, s) for c, s in zip(sortedcandidates, sortedscores) if s[0] < th]))
    # if len(tup) == 0:
    #     return Node(name,frame,[],[])
    # node = Node(name,frame,tup[0],tup[1])
    node = Node(name,frame,sortedcandidates,sortedscores)
    return node

