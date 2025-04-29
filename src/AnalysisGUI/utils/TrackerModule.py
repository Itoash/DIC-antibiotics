#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:40:26 2025

@author: victorionescu
"""
# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from numba import jit
import concurrent.futures
import time
from scipy.spatial import distance_matrix
from skimage.draw import polygon, line
from copy import deepcopy
from numba import njit
import os

def get_warp(img1, img2, motion=cv2.MOTION_TRANSLATION):
    """Compute the warp matrix between two images."""
    imga = img1.copy().astype(np.float32)
    imgb = img2.copy().astype(np.float32)
    if len(imga.shape) == 3:
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    if len(imgb.shape) == 3:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warpMatrix = np.eye(3, 3, dtype=np.float32)
    else:
        warpMatrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix = cv2.findTransformECC(templateImage=imga, inputImage=imgb,
                                       warpMatrix=warpMatrix, motionType=motion)[1]
    return warp_matrix

def create_warp_stack(imgs):
    """Create a stack of warp matrices for consecutive image pairs."""
    warp_stack = []
    # Handle the case when there's only one image
    if len(imgs) <= 1:
        return np.array(warp_stack)  # Return empty array
    
    for i, img in enumerate(imgs[:-1]):
        warp_stack.append(get_warp(img, imgs[i+1]))
    return np.array(warp_stack)

def get_border_pads(img_shape, warp_stack):
    """Calculate padding needed to accommodate all warped images."""
    if len(warp_stack) == 0:
        print("No warpstack - using default padding")
        return 0, 0, 0, 0  # Return zeros for top, bottom, left, right
    
    maxmin = []
    corners = np.array([[0, 0, 1], [img_shape[1], 0, 1], 
                        [0, img_shape[0], 1], [img_shape[1], img_shape[0], 1]]).T
    warp_prev = np.eye(3)
    
    for warp in warp_stack:
        # Ensure warp is 3x3 for matrix multiplication
        if warp.shape[0] == 2:  # If 2x3 matrix
            warp = np.concatenate([warp, [[0, 0, 1]]])
        warp = np.matmul(warp, warp_prev)
        warp_invs = np.linalg.inv(warp)
        new_corners = np.matmul(warp_invs, corners)
        xmax, xmin = new_corners[0].max(), new_corners[0].min()
        ymax, ymin = new_corners[1].max(), new_corners[1].min()
        maxmin.append([ymax, xmin])
        maxmin.append([ymin, xmax])
        warp_prev = warp.copy()
    
    if not maxmin:  # Safeguard in case the loop doesn't run
        return 0, 0, 0, 0
        
    maxmin = np.array(maxmin)
    bottom = max(0, int(maxmin[:, 0].max()))
    top = min(0, int(maxmin[:, 0].min()))
    left = min(0, int(maxmin[:, 1].min()))
    right = max(0, int(maxmin[:, 1].max()))
    top_pad = max(0, int(-top))
    bottom_pad = max(0, int(bottom-img_shape[0]))
    left_pad = max(0, int(-left))
    right_pad = max(0, int(right-img_shape[1]))
    
    return top_pad, bottom_pad, left_pad, right_pad

def homography_gen(warp_stack):
    """Generator that yields the cumulative homography matrices."""
    # Handle empty warp_stack
    if len(warp_stack) == 0:
        return
        
    H_tot = np.eye(3)
    
    # Create 3x3 homography matrices from the 2x3 or 3x3 warp matrices
    wsp = []
    for warp in warp_stack:
        if warp.shape[0] == 2:  # If 2x3 matrix
            wsp.append(np.vstack([warp, [0, 0, 1]]))
        else:  # Already 3x3
            wsp.append(warp)
    
    for i in range(len(wsp)):
        H_tot = np.matmul(wsp[i], H_tot)
        yield np.linalg.inv(H_tot)

def apply_warping_fullview(images, warp_stack=None, PATH=None):
    """Apply warping to create stabilized images with full view."""
    # Handle the case when there's only one image
    if len(images) <= 1:
        return images
    
    # If warp_stack is None, create it
    if warp_stack is None:
        warp_stack = create_warp_stack(images)
    
    # If warp_stack is still empty (e.g., single image), return original images
    if len(warp_stack) == 0:
        return images
    
    top, bottom, left, right = get_border_pads(
        img_shape=images[0].shape, warp_stack=warp_stack)
    
    # Check if any dimension would be zero
    if images[0].shape[0] + top + bottom <= 0 or images[0].shape[1] + left + right <= 0:
        print("Warning: Calculated padding would result in zero-sized image. Using minimal padding.")
        top = bottom = left = right = 1  # Use minimal padding
    
    # Create padded first image with non-zero dimensions
    padded_height = max(1, images[0].shape[0] + top + bottom)
    padded_width = max(1, images[0].shape[1] + left + right)
    image_0 = np.zeros((padded_height, padded_width))
    
    # Careful slicing to avoid empty dimensions
    bottom_slice = None if bottom == 0 else -bottom
    right_slice = None if right == 0 else -right
    
    # Place the original image into the padded area
    if top < padded_height and left < padded_width:
        target_height = min(images[0].shape[0], padded_height - top)
        target_width = min(images[0].shape[1], padded_width - left)
        
        if target_height > 0 and target_width > 0:
            image_0[top:top+target_height, left:left+target_width] = images[0][:target_height, :target_width]
    
    imgs = [image_0]
    H_generator = homography_gen(warp_stack)
    try:
        for i, img in enumerate(images[1:]):
            try:
                H_tot = next(H_generator) + np.array([[0, 0, left], [0, 0, top], [0, 0, 0]])
                img_warp = cv2.warpPerspective(
                    img, H_tot, (img.shape[1]+left+right, img.shape[0]+top+bottom), 
                    flags=cv2.INTER_NEAREST)
                
                if PATH is not None:
                    filename = PATH + "".join([str(0)]*(3-len(str(i)))) + str(i) + '.png'
                    cv2.imwrite(filename, img_warp)
                
                imgs.append(img_warp)
            except StopIteration:
                print(f"Warning: Not enough homography matrices for image {i+1}")
                # Just add the original image with padding
                padded_img = np.zeros((images[0].shape[0]+top+bottom, images[0].shape[1]+left+right))
                padded_img[top:top+img.shape[0], left:left+img.shape[1]] = img
                imgs.append(padded_img)
    except Exception as e:
        print(f"Error during warping: {str(e)}")
    
    return imgs


@njit
def filterframe(image: np.ndarray, col: int):
    dims = image.shape
    filtered = image.flatten()
    for i in range(len(filtered)):
        if filtered[i] != 1.0:
            filtered[i] = 0
    return filtered.reshape(dims).T

    # more stuff needed to implement in this function:
    # continuity
    # boundary conditions
    # union of most likely candidates by overlap
    # separation of candidates based on nodes
    # drawing in missing cells
    # discarding garbage at the end


class basicplot:  # fucked up GUI
    def __init__(self, image, colpositions, colareas, index):
        print("Loaded image "+str(index)+"of filetype ")
        self.image = image
        self.fig = plt.figure()
        plt.ion()
        plt.imshow(image)
        plt.show(block=False)
        self.previmage = image
        self.colpositions = colpositions
        self.prevpos = colpositions
        self.colareas = colpositions
        self.prevarea = colareas
        self.imgindex = index
        self.mode = 'CLI'
        self.selectedcolor = None
        self.returnvalue = self.image
        self.drawing = False  # Track if we are currently drawing
        self.disabledrawing = False
        self.disableerasing = False
        self.xs = []  # List to store x coordinates of the drawn points
        self.ys = []

    def mouse_event(self, event):
        print('\n x: {} , y: {},imvalue {}'.format(event.xdata,
              event.ydata, self.image[int(event.ydata), int(event.xdata)]))

    def undo(self):
        print("Undid")
        self.image = self.previmage
        self.colpositions = self.prevpos
        self.colareas = self.prevarea
        if isinstance(self.image, np.ndarray):
            if np.sum(self.image-self.previmage) == 0:

                print("Did nothing")

    def updateimage(self, newvalue):
        self.previmage = self.image.copy()
        self.image = newvalue

    def updatedata(self, area, pos):
        self.prevpos = self.colpositions
        self.colpositions = pos
        self.prevarea = self.colareas
        self.colareas = area

    def mouse_event_erase(self, event):
        if event.inaxes is None or self.disableerasing:
            return  # Ignore mouse events outside the axes
        color = self.image[int(event.ydata), int(event.xdata)]

        dummyimage = self.image.copy()
        dummyimage[self.image == float(color)] = 0
        dummyimage[self.image > float(color)] -= 1
        a = self.colareas.copy()
        p = self.colpositions.copy()
        if not color > len(self.colareas):
            del a[int(color)-1]
            del p[int(color)-1]
            self.updatedata(a, p)

        self.updateimage(dummyimage)

        plt.imshow(self.image)
        plt.draw()

    def mouse_event_fill(self, event):
        print("Set to drawfill mode")

    def mouse_event_pick(self, event):
        print("Set to pick mode")

    def CLImode(self):
        self.disabledrawing = True
        self.disableerasing = True
        plt.imshow(self.image)
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_event)
        plt.draw()
        print("---Frame: "+str(self.imgindex)+"---")
        for col in range(1, int(np.max(self.image))+1):
            print("Label ID: "+str(col))
            print("Position: "+str(self.colpositions[col-1]))

            print("Area: "+str(self.colareas[col-1]))
            print('\n')
        userinput = input(
            "Enter label(s) to be deleted (separated by commas), or elimination criteria (syntax: a/x/y </> value): ")
        # make a switch case

        match userinput:
            # exit if called
            case userinput if 'xit' in str(userinput):
                self.updateimage(None)
                return self.image
            # assume you have a list of labels
            case userinput if ',' in str(userinput):
                labelarray = userinput.split(',')
                labelarray = [int(el) for el in labelarray]
                labelarray = list(sorted(labelarray))
                dummyimage = self.image.copy()
                for i, el in enumerate(labelarray):
                    dummyimage[self.image == float(el)] = 0
                    dummyimage[self.image > float(el)] -= 1
                    a = self.colareas.copy()
                    p = self.colpositions.copy()
                    del a[int(el)-1-i]
                    del p[int(el)-1-i]
                self.updatedata(a, p)
                self.updateimage(dummyimage)
                self.CLImode()
                return
            case userinput if 'a<' in str(userinput) or 'a>' in str(userinput):
                symbol = userinput[1]
                limit = float(userinput[2:])
                if symbol == '>':

                    labelarray = [self.colareas.index(
                        i)+1 for i in self.colareas if i > limit]
                    print("labels to delete: "+str(labelarray))
                elif symbol == '<':
                    labelarray = [self.colareas.index(
                        i)+1 for i in self.colareas if i < limit]
                    print("labels to delete: "+str(labelarray))
                dummyimage = self.image.copy()

                for i, el in enumerate(labelarray):
                    dummyimage[self.image == float(el)] = 0
                    dummyimage[self.image > float(el)] -= 1
                    a = self.colareas.copy()
                    p = self.colpositions.copy()
                    del a[int(el)-1-i]
                    del p[int(el)-1-i]
                self.updatedata(a, p)
                self.updateimage(dummyimage)
                self.CLImode()

            case userinput if userinput == 'r':
                self.undo()
                self.CLImode()

            case userinput if len(userinput) == 1 and int(userinput) == 0:

                return self.image

            case userinput if str(userinput) in ['CLI', 'draw', 'erase', 'fill']:
                self.mode = userinput
                if self.mode == 'CLI':
                    self.CLImode()
                if self.mode == 'draw':
                    self.drawmode()
                if self.mode == 'erase':
                    self.erasemode()

            case _:
                try:
                    dummyimage = self.image.copy()
                    dummyimage[self.image == float(userinput)] = 0
                    dummyimage[self.image > float(userinput)] -= 1
                    self.updateimage(dummyimage)
                    a = self.colareas.copy()
                    p = self.colpositions.copy()
                    del a[int(userinput)-1]
                    del p[int(userinput)-1]
                    self.updatedata(a, p)
                    self.CLImode()
                except ValueError:
                    print("Invalid input, restarting sequence")
                    self.CLImode()

    def mouse_event_draw(self, event):
        if event.inaxes is None or self.selectedcolor is None or self.disabledrawing:
            return  # Ignore mouse events outside the axes

        if event.button == 1:  # Left mouse button clicked
            if not self.drawing:  # Start drawing
                self.drawing = True
                # Start new drawing at the click position
                self.xs = [event.xdata]
                self.ys = [event.ydata]
            else:  # Continue drawing while dragging
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.update_drawing()
        # Right mouse button clicked (optional, to stop drawing)
        elif event.button == 3:
            self.drawing = False
            print("Stopped drawing")

    def update_drawing(self):
        if self.drawing:
            polygonx, polygony = polygon(np.asarray(self.xs).astype(
                int), np.asarray(self.ys).astype(int))

            # Update the plot with the drawn line
            dummy = self.image

            dummy[polygony, polygonx] = self.selectedcolor

            self.updateimage(dummy)

           # Redraw the plot to show the updated drawing

    def mouse_event_release(self, event):
        if self.drawing:  # End drawing when mouse button is released
            self.drawing = False

            print(f"Finished drawing with points: {len(self.xs)}")
            plt.imshow(self.image)
            plt.draw()
            self.xs = []
            self.ys = []
            # You can update the image here based on your drawing (e.g., by adding shapes to the image)

    def toggledrawing(self):
        self.disabledrawing = not self.disabledrawing

    def toggleerasing(self):
        self.disableerasing = not self.disableerasing

    def drawmode(self):
        self.disabledrawing = False
        self.disableerasing = True

        plt.imshow(self.image)
        self.fig.canvas.mpl_connect(
            'button_press_event', self.mouse_event_draw)
        self.fig.canvas.mpl_connect(
            'motion_notify_event', self.mouse_event_draw)
        self.fig.canvas.mpl_connect(
            'button_release_event', self.mouse_event_release)

        plt.draw()
        if self.disabledrawing:
            print("Drawing is disabled.")
        elif not self.disabledrawing:
            print("Warning: drawing is enabled.")
        userinput = input(
            "Select a color/lw , or press 0 to exit,or p to pause/unpause drawing: ")
        match userinput:
            case userinput if str(userinput) in ['CLI', 'draw', 'erase', 'fill']:

                self.mode = userinput
                if self.mode == 'CLI':
                    self.CLImode()
                if self.mode == 'draw':
                    self.drawmode()
                if self.mode == 'erase':
                    self.erasemode()
            case userinput if userinput == '0':
                return
            case userinput if 'xit' in userinput:
                self.updateimage(None)
                return
            case userinput if userinput == 'p':
                self.toggledrawing()
                self.drawmode()
            case userinput if ',' in userinput:
                self.selectedcolor = int(userinput.split(',')[0])
                self.selectedwidth = int(userinput.split(',')[1])
                self.drawmode()
            case userinput if userinput == 'r':
                self.undo()
                self.drawmode()
            case _:
                print("Invalid input, try again")
                self.drawmode()

    def erasemode(self):
        self.disabledrawing = True
        self.disableerasing = False
        plt.imshow(self.image)
        self.fig.canvas.mpl_connect(
            'button_press_event', self.mouse_event_erase)

        plt.draw()
        if self.disableerasing:
            print("Erasing is disabled.")
        elif not self.disableerasing:
            print("Warning: Erasing is enabled.")
        userinput = input("Press 0 to exit,or p to pause/unpause erasing: ")
        match userinput:
            case userinput if str(userinput) in ['CLI', 'draw', 'erase', 'fill']:
                self.mode = userinput
                if self.mode == 'CLI':
                    self.CLImode()
                if self.mode == 'draw':
                    self.drawmode()
                if self.mode == 'erase':
                    self.erasemode()
            case userinput if userinput == '0':
                return
            case userinput if 'xit' in userinput:
                self.image = None
                return
            case userinput if userinput == 'p':
                self.toggleerasing()
                self.erasemode()
            case userinput if userinput == 'r':
                self.undo()
                self.erasemode()
            case _:
                print("Invalid input, try again")
                self.erasemodel()

    def main(self):
        print("Image type is "+str(type(self.image)))
        if not isinstance(self.image, np.ndarray):
            print("returning none in main")
            return None

        self.CLImode()

        return self.image


def printimgswithdata2(imagestack, framelist):
    newimagestack = []
    for index, i in enumerate(imagestack):
        framedata = framelist[index]
        comlist = [framedata['COMs'][el] for el in framedata['COMs'].keys()]
        arealist = [framedata['areas'][el] for el in framedata['areas'].keys()]
        print(type(i))
        try:
            print(f"Creating basicplot object for index {index}")
            plot = basicplot(i, comlist, arealist, index)
            newi = plot.main()
        except Exception as e:
            print(f"Error while creating basicplot for index {index}: {e}")

        if newi is None:
            dif = len(imagestack)-len(newimagestack)
            newimagestack += imagestack[dif:]
            break
        newimagestack.append(newi)
    return newimagestack


# Conditions to satisfy for tracking:
    # 1. non-zero overlap
    # 2. at least 20% of area (considering top 4 candidates)
    # 3. closeness
    # 4. ellipse angle ratio
    # 5. Ellipse major ratio
# Construct evaluate_candidates function:
    # check top 4 most overlap
    # check area ratio
    # Check angle ratio
    # Create cost function: has to be more than 2 out of 4
    # choose top 2 candidates
def checkifunder(array, name):
    if np.any(array > 1):
        print("Problem in "+str(name)+"!")
        print("Found value((s) "+str(np.where(array > 1)) +
              " at positions "+str(np.argwhere(array > 1)))


def evaluatecandidates(distancetoothers, overlaps, anglestoothers, areatoothers, arstoothers, name, frame, minoverlap=0.15, th=0, weights=[1, 1, 1, 1, 1]):
    # distancetoothers: 1D array in label order
    # overlaps: 2D array with labels in frst dim and overlaps in second dim
    # anglestoothers: 1D array with angle ratio
    # areatoothers: 1D array with area ratio

    # if it has substantial overlaps in next frame, continue scoring
    if isinstance(overlaps, np.ndarray):
        indexes = np.argwhere(overlaps[1, :] > minoverlap)
        if len(indexes) == 0:  # no significant overlap
            
            return Node(name, frame, [], [])
        candidates = overlaps[0, indexes].astype(int)-1
        candidateoverlaps = overlaps[1, indexes]*weights[1]

        # double overlap score, then set larger than one values to max to account for half overlaps
        candidatehalfoverlaps = candidateoverlaps*2
        candidatehalfoverlaps[candidatehalfoverlaps > 1] = 1

    else:  # otherwise return None
        
        return Node(name, frame, [], [])
    # assign no-division scores
    candidateangles = anglestoothers[candidates]*weights[2]

    candidateareas = areatoothers[candidates]*weights[3]
    candidatears = arstoothers[candidates]*weights[4]
    candidatedistances = distancetoothers[candidates]*weights[0]
    candidatedistances[candidatedistances > 1] = 1

    # add half-area score for good candidates
    candidatehalfareas = candidateareas*2
    candidatehalfareas[candidatehalfareas > 1] = 1
    # add half aspect ratio score
    candidatehalfars = candidatears*2
    candidatehalfars[candidatehalfars > 1] = 1
    # add up scores; max is 6, anything above a 3 is pretty good with image stabilization on
    candidatescores = (candidateoverlaps+candidateangles+candidateareas+candidatedistances+candidatehalfoverlaps +
                       candidatehalfareas+candidatears+candidatehalfars)/(np.sum(weights)+weights[3]+weights[4]+weights[1])
    candidates += 1
    finalcandidatescores = candidatescores.tolist()
    finalcandidates = candidates.tolist()

    _, sortedcandidates = list(
        zip(*sorted(zip(finalcandidatescores, finalcandidates))))
    sortedscores = list(sorted(finalcandidatescores))
    # tup = tuple(zip(*[
    # (c, s) for c, s in zip(sortedcandidates, sortedscores) if s[0] < th]))
    # if len(tup) == 0:
    #     return Node(name,frame,[],[])
    # node = Node(name,frame,tup[0],tup[1])
    node = Node(name, frame, sortedcandidates, sortedscores)
    
    return node




def filterbyoverlap(image, overlapdict, plotflag=False):

    eliminatedlabels = []
    for k in overlapdict.keys():

        if isinstance(overlapdict[k], np.ndarray):
            if np.sum(overlapdict[k][1, :]) < 0.5:
                eliminatedlabels.append(k)
        else:
            eliminatedlabels.append(k)
    duplicate = image.copy()
    for no in eliminatedlabels:
        duplicate[duplicate == float(no)] = 0
    duplicate = np.reshape(duplicate, np.shape(image))
    if plotflag == True:
        fig, ax = plt.subplots(3)
        ax[0].imshow(image)
        ax[1].imshow(duplicate)
        ax[2].imshow(image-duplicate)
        plt.show(block=False)
    return duplicate


def getgoodnodes(nodelist, startingnodes=None):
    nodes = nodelist.copy()
    if startingnodes is None:
        goodnodes = [n.name for n in nodes[0]]
    else:
        goodnodes = startingnodes
    newnodes = []
    for f in nodes:
        copy = [n for n in f if n.name in goodnodes]
        goodnodes = []
        for n in copy:
            goodnodes += n.children.keys()

        newnodes.append(copy)
    return newnodes


def filterimagebynodes(imagestack, nodelist):
    imstack = []
    for i in range(len(imagestack)):
        img = imagestack[i].copy()
        values = np.arange(1, np.max(img)+1)
        allowed = [n.name for n in nodelist[i]]
        forbidden = [el for el in values if el not in allowed]

        for c in forbidden:
            img[img == c] = 0
        imstack.append(img)

    return imstack

def stabilize_images(labels,DCs,ACs = None):
    warp_stack = create_warp_stack(labels)
    stable_labels = apply_warping_fullview(labels, warp_stack)
    
    stable_DCs = apply_warping_fullview(DCs, warp_stack)
    if ACs is not None:
        stable_ACs = apply_warping_fullview(ACs, warp_stack)
        return stable_labels,stable_DCs,stable_ACs
    else:
        return stable_labels,stable_DCs
    
def main():
    # keys for framedata: position = 0;area = 1;angle = 2;ar = 3;overlap = 4
    path = '/Users/victorionescu/Desktop/Tracker test'
    filenames = [path+"/"+f for f in listdir(path) if f != ".DS_Store"]
    sorted_filenames = sorted(filenames, key=lambda x: int(x[-8:-4]))
    imgs = [cv2.imread(f, -1) for f in sorted_filenames[:]]
    print(len(imgs))
    warpstack = create_warp_stack(imgs)

    processedimgs = apply_warping_fullview(imgs, warpstack)

    processedimgs = remaplabels(processedimgs)

    # f = filterframe(processedimgs[0].copy())
    stackdata, dist = multiprocessframes(processedimgs)

    tracklist = []
    stackdata.append(stackdata[-1])
    for i in range(len(dist)):
        tracklist.append(evaluateframe(
            processedimgs[i], stackdata[i], stackdata[i+1], dist[i], i))
    stackdata = stackdata[:-1]
    newnodes = tracklist.copy()

    nuimg = filterimagebynodes(processedimgs[:-1], newnodes)
    network = imagenetwork(processedimgs, tracklist, stackdata)
    return stackdata, dist, tracklist, newnodes, network

    
def obtain_network(imgs, scorethreshold=0.5):
    # imgs: list of 2D numpy arrays as images
    # scorethreshold: double to set in imagenetwork class
    processedimgs = imgs.copy()
    for idx,i in enumerate(processedimgs):
        if not np.any(i):
            return idx
    stackdata, dist = multiprocessframes(processedimgs)


    stackdata.append(stackdata[-1])
    tracklist = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count()-1) as executor:
        tracklist = list(executor.map(evaluateframe,processedimgs,stackdata[:-1],stackdata[1:],dist,[i for i in range(0,len(dist),1)]))
    # tracklist = []
    # for i in range(len(dist)):
    #     tracklist.append(evaluateframe(
    #         processedimgs[i], stackdata[i], stackdata[i+1], dist[i], i))
    
    stackdata = stackdata[:-1]
    network = imagenetwork(processedimgs, tracklist,
                            stackdata, scorethreshold=scorethreshold)
    
    return network


def remaplabels(stack):
    for i, img in enumerate(stack):
        if len(np.unique(img)) != np.max(stack[i])+1:
            for i, label in enumerate(np.unique(img)):
                img[img == label] = i

    return stack


def multiprocessframes(imagestack):

    opstack = imagestack.copy()
    # zero-pad stack to add data about last framw w/o overlaps
    opstack.append(np.zeros_like(opstack[0]))
    with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
        stackdata = list(executor.map(
            process_frame, opstack[:-1], opstack[1:], np.arange(0, len(imagestack))))
    # stackdata = []
    # for prev,nex,idx in zip(opstack[:-1],opstack[1:],[i for i in range(len(imagestack))]):
    #     stackdata.append(process_frame(prev,nex,idx))
    COMlist = []
    for el in stackdata:
        COMs = [i for i in el[0]]
        COMs = np.asarray(COMs)
        COMlist.append(COMs)
    COMlist.append(COMlist[-1])
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        dist = list(executor.map(inversedistancematrix,
                    COMlist[:-1], COMlist[1:]))
    return stackdata, dist


def process_frame(currentimage, nextimage, imageid):
    overlapdict = {}
    COMdict = {}
    angledict = {}
    majordict = {}
    minordict = {}
    areadict = {}
    COMlist = []
    arealist = []
    angleslist = []
    overlaplist = []
    arlist = []
    majorlist = []
    if len(np.unique(currentimage)) != np.max(currentimage)+1:
        for i, label in enumerate(np.unique(currentimage)):
            currentimage[currentimage == label] = i
    if len(np.unique(nextimage)) != np.max(nextimage)+1:
        for i, label in enumerate(np.unique(nextimage)):
            nextimage[nextimage == label] = i
    for col in range(1, int(np.max(currentimage))+1):
        dims = np.shape(currentimage)

        filtered = currentimage.copy()
        filtered.flatten('F')
        filtered[filtered != float(col)] = 0
        filtered = filtered/col
        filtered.astype(float)
        filtered = np.reshape(filtered, dims)
        # coordinates for all points
        coords = np.argwhere(filtered == 1)
        # centroid coords
        com = np.mean(coords, axis=0)

        area = np.sum(filtered)

        # ellipse angle?

        if np.sum(filtered) > 1:

            contour, hierarchy = cv2.findContours(filtered.astype(
                'uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = contour[0]

            cdims = np.shape(contour)
            contour = np.reshape(contour, (cdims[0], cdims[2]))
            if len(contour[:, 0]) > 4:

                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                major = ellipse[1][1]
                minor = ellipse[1][0]
            else:
                angle = 0
                major = 1
                minor = 1
            overlaps = np.multiply(filtered, nextimage)
            counts = np.unique(overlaps, return_counts=True)
            counts = np.asarray(counts)
            if len(counts[0, :]) > 1:
                counts = counts[:, 1:]
                counts[1, :] /= np.sum(filtered)
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

    return (np.asarray(COMlist), np.asarray(arealist), np.asarray(angleslist), np.asarray(arlist), overlaplist, np.asarray(majorlist))


def distancematrix(COMarray1, COMarray2):
    dists = distance_matrix(COMarray1, COMarray2)
    return dists


def inversedistancematrix(COMarray1, COMarray2):
    dists = distance_matrix(COMarray1, COMarray2)
    dists[dists == 0] = 0.0001
    dists = 1/dists
    dists[dists > 1] = 1
    return dists


def ratiomatrix(array1, array2):
    matrix = np.empty((len(array1), len(array2)))

    for i in range(len(array1)):
        matrix[i, :] = array2/array1[i]

    matrix = matrix.flatten()

    largerthanone = np.argwhere(matrix > 1)
    for i in largerthanone:
        matrix[i] = 1/matrix[i]
    matrix = np.reshape(matrix, (len(array1), len(array2)))

    # ratio of every element in array1 (first dim) to every element in array2(second dim)
    return matrix


def evaluateframe(frame, framedict1, framedict2, distancematrix, ind):
    
    distancelist = [distancematrix[i, :]
                    for i in range(distancematrix.shape[0])]
    angleslist3 = [i for i in framedict1[2]]

    angleslist4 = [i for i in framedict2[2]]
    arealist3 = [i for i in framedict1[1]]
    arealist4 = [i for i in framedict2[1]]
    arlist3 = [i for i in framedict1[3]]
    arlist4 = [i for i in framedict2[3]]
    overlaplist = [i for i in framedict1[4]]
    anglematrix = ratiomatrix(np.asarray(angleslist3), np.asarray(angleslist4))
    anglematrix = [anglematrix[i, :] for i in range(anglematrix.shape[0])]
    areamatrix = ratiomatrix(np.asarray(arealist3), np.asarray(arealist4))
    areamatrix = [areamatrix[i, :] for i in range(areamatrix.shape[0])]
    armatrix = ratiomatrix(np.asarray(arlist3), np.asarray(arlist4))
    armatrix = [armatrix[i, :] for i in range(armatrix.shape[0])]
    frametrack = []
    for i in range(len(distancelist)):
        frametrack.append(evaluatecandidates(distancelist[i], overlaplist[i], anglematrix[i], areamatrix[i],
                          armatrix[i], i+1, ind))
    # with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
    #     frametrack = list(exe.map(evaluatecandidates, distancelist, overlaplist, anglematrix, areamatrix,
    #                       armatrix, np.arange(1, np.max(frame)+1), [ind for i in range(len(distancelist))]))
        # index 0 is label 1, and so on
    return frametrack


class Node:

    def __init__(self, colorname, frame, childrenlist, weightlist):
        self.name = colorname
        self.frame = frame
        self.likelychildrenweights = {}
        self.likelychildrenweights = {c[0]: w[0]
                                      for c, w in zip(childrenlist, weightlist)}
        self.tbd = False
        self.position = None
        self.area = None
        self.angle = None
        self.overlaps = None
        self.major = None
        self.minor = None
        self.likelychildren = []
        self.vcoords = []
        self.metadata = {'colname': self.name, 'frame': self.frame}
        self.convexity = 1
        self.likelyparent = []
        self.backwardsoverlaps = None
        self.trackname = None
        
    def addchild(self, child, weight):
        if child not in self.children.keys():
            self.children.update({child: weight})

    def printlineage(self, plot=True):
        for c, w in zip(self.children.keys(), self.children.items()):
            plt.plot([self.frame, self.frame+1],
                     [self.name, c], c='k', ms=w[1])
            plt.scatter([self.frame, self.frame+1],
                        [self.name, c], c='k', s=w[1])

        def __str__(self):
            return "Label: "+str(self.name) + "; Frame: " + str(self.frame)+"; TBD = "+str(self.tbd)

    def __hash__(self):

        return hash(str(self))

    def __eq__(self, other):
        return self.name == other.name and self.frame == other.frame

    def removeNode(self, node, debug=False):
        name = node.name
        # check if node is in nextframe
        if node.frame == self.frame+1:
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
            # get coords of name, and check that it is not empty, and overlaps is not None
            if self.overlaps is not None:
                coords = np.argwhere(self.overlaps == name)
                if len(coords) != 0:  # always 2D array, so need second dimension

                    self.overlaps = np.delete(
                        self.overlaps, (coords[0, 1]), axis=1)
                coords = np.argwhere(self.overlaps[0, :] > name)
                if len(coords) != 0:
                    # always 2D array, so need second dimension

                    self.overlaps[0, coords] -= 1
            if debug:
                print("Target node removed. Printing data...")
                print(f'Name: {self.name},Frame: {self.frame};\n Overlapdict: {self.likelychildrenweights}\n Overlaps: {self.overlaps} \n Backoverlaps: {self.backwardsoverlaps}')
        if node.frame == self.frame-1:
            if debug:
                print("Target node is in previous frame. Printing current data...")
                print(f'Name: {self.name},Frame: {self.frame};\n Overlapdict: {self.likelychildrenweights}\n Overlaps: {self.overlaps} \n Backoverlaps: {self.backwardsoverlaps}')

            if self.backwardsoverlaps is not None:
                # always 2D array, so need second dimension
                coords = np.argwhere(self.backwardsoverlaps == name)
                if len(coords) != 0:

                    self.backwardsoverlaps = np.delete(
                        self.backwardsoverlaps, (coords[0, 1]), axis=1)
                # always 2D array, so need second dimension
                coords = np.argwhere(self.backwardsoverlaps[0, :] > name)
                if len(coords) != 0:

                    self.backwardsoverlaps[0, coords] -= 1
            if debug:
                print("Target node removed. Printing data...")
                print(f'Name: {self.name},Frame: {self.frame};\n Overlapdict: {self.likelychildrenweights}\n Overlaps: {self.overlaps} \n Backoverlaps: {self.backwardsoverlaps}')
        if node.frame == self.frame and self.name > node.name:
            print(f"Target is in same frame ({self.name},{self.frame})")
            self.name -= 1


class imagenetwork:
    def __init__(self, images, nodes, stackdata, scorethreshold=0.3):
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
        copy = []
        for nlist in nodes:
            copy += nlist
        self.nodes = copy
        self.similarity_threshold = 0.3
        self.mt_getbackoverlaps()
        self.mt_findchildren()
        self.makeTree()  # uncomment when sure that it makes a tree and not a graph
        self.obtainLineages()
        self.virtualizeCoords()
        self.mt_findconvexity()

    # have to make methods for:
        # - patching up missing labels - easy enough, look forwards
        # - removing garbage - done, can reliably remove nodes and labels
        # - splitting undersegmented stuff - harder, if time gap is not too large can split based on boundary, then recompute overlaps
        # - filling in oversegmented stuff - even harder, have to detect when area decreases suddenly and multiple bad overlaps appear
        # - determining network - have to have flags for: out of frame, lost contrast etc.

    def reassignLabels(self,oldlabels,newlabels,frame_idx):
        # relevant_nodes = [n for n in self.nodes if n.frame == frame_idx]
        # for n in relevant_nodes:
        #     if n.name in oldlabels:
        #         idx = np.argwhere(oldlabels == n.name)
        #         n.name = newlabels[idx]
        #     else:
        #         print("Could not find node when remapping, node has disappeared")
        pass
                
    def mt_findchildren(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.findchildren, self.nodes)

    def mt_getbackoverlaps(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.getbackoverlaps, np.arange(1, len(self.images)))

    def findchildren(self, n):
        children = [el[0] for el in n.likelychildrenweights.items()]
        weights = [el[1] for el in n.likelychildrenweights.items()]
        childrenlist = [el for el, w in zip(
            children, weights) if w > self.similarity_threshold]
        n.likelychildren = [
            el for el in self.nodes if el.name in childrenlist and el.frame == n.frame+1]
        for node in n.likelychildren:
            node.likelyparent.append(n)
            
    def mt_findconvexity(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.findconvexity, np.arange(0, len(self.images)))
    def findconvexity(self,i):
        img = self.images[i]
        
        nodes = [n for n in self.nodes if int(n.frame) == int(i)]
        
        for node in nodes:
            col = node.name
            mask = (img.astype(float) == float(col))
            filtered = np.zeros_like(img, dtype=float)
            filtered[mask] = 1
            contour,_ = cv2.findContours(
                filtered.astype('uint8'),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            contour = [contour[0]]
            convex_contour = cv2.convexHull(contour[0])
            convex_image =np.zeros_like(filtered)
            convex_image = cv2.drawContours(convex_image,[convex_contour],0,color = 1,thickness = -1)
            convex_area = np.count_nonzero(convex_image.flatten())
            original_area = np.count_nonzero(filtered.flatten())
            node.convexity = original_area/convex_area
            

    def makeTree(self):  # subject to change, may be a bit overconfident in doing this

        for node in self.nodes:
            if len(node.likelyparent) > 1:
                # choose parent with highest overlap
                best_parent_index = np.argmax(node.backwardsoverlaps[1, :])
                best_parent_name = node.backwardsoverlaps[0, best_parent_index]
                badparents = [
                    n for n in node.likelyparent if n.name != best_parent_name]
                for p in badparents:
                    p.likelychildren.remove(node)
                node.likelyparent = [
                    n for n in node.likelyparent if n.name == best_parent_name]

    def debug(self):
        for i in range(len(self.images)):
            print(np.unique(self.images[i]))
            print([n.name for n in self.nodes if n.frame == i])

    def getbackoverlaps(self, index):

        for col in range(1, int(np.max(self.images[index]+1))):

            node = [n for n in self.nodes if n.name ==
                    col and n.frame == float(index)]
            try:
                node = node[0]
            except IndexError:
                print(node, index, col)
            dims = np.shape(self.images[index])

            filtered = self.images[index].copy()
            filtered.flatten('F')
            filtered[filtered != float(col)] = 0
            filtered = filtered/col
            filtered.astype(float)
            filtered = np.reshape(filtered, dims)
            overlaps = np.multiply(filtered, self.images[index-1])
            counts = np.unique(overlaps, return_counts=True)
            counts = np.asarray(counts)
            if len(counts[0, :]) > 1:
                counts = counts[:, 1:]
                counts[1, :] /= np.sum(filtered)
            else:
                counts = None
            node.backwardsoverlaps = counts

    def deletenode(self, name, frame):
        chosenimage = self.images[frame]
        chosennode = [n for n in self.nodes if n.name ==
                      name and n.frame == frame][0]
        chosenimage[chosenimage == name] = 0
        chosenimage[chosenimage > name] -= 1
        chosennode.tbd = True
        for n in self.nodes:
            n.removeNode(chosennode)
        self.nodes.remove(chosennode)
        del chosennode

    def unlinknode(self, nodepos, direction):
        node = [n for n in self.nodes if n.name ==
                nodepos[1] and n.frame == nodepos[0]][0]
        if direction.lower() == 'r':
            if len(node.likelychildren)>0:
                for ch in node.likelychildren:
                    ch.likelyparent = []
            node.likelychildren = []
        elif direction.lower() == 'l':
            if len(node.likelyparent)>0:
                for pt in node.likelyparent:
                    pt.likelychildren.remove(node)
                node.likelyparent = []
        self.makeTree()
        self.virtualizeCoords()
        self.obtainLineages()
        
    def main(self):
        return self

    def exportGraph(self):
        nodes_pos = np.zeros((len(self.nodes), 2))
        edge_indexes = []
        indexmap = {}
        texts = []
        meta = []
        # fill up array with nodes
        for i, n in enumerate(self.nodes):
            nodes_pos[i, 1] = n.vcoords[1]
            nodes_pos[i, 0] = n.vcoords[0]
            indexmap.update({n: i})
            t = n.trackname
            texts.append(t)
            meta.append((n.frame, n.name,n.area,n.convexity,n.major/n.minor))
        for n in self.nodes:
            nodeindex = indexmap[n]
            for ch in n.likelychildren:
                chindex = indexmap[ch]
                edge_indexes.append((nodeindex, chindex))

        edge_indexes = np.asarray(edge_indexes)
        return nodes_pos, edge_indexes, texts, meta

    def traverseTree(self, node, occupiedlist):
        if len(node.vcoords) == 0:
            lowest_y = np.max(occupiedlist[node.frame])
            node.vcoords = [node.frame, lowest_y+1]
            occupiedlist[node.frame].append(lowest_y+1)
        for ch in node.likelychildren:
            self.traverseTree(ch, occupiedlist)

    def traverseLineage(self, start, name, dictionary):
        node = start
        lineagedict = {}
        while (len(node.likelychildren) == 1):
            lineagedict.update({node.frame: node.name})
            node.trackname = name
            node = node.likelychildren[0]
        lineagedict.update({node.frame: node.name})
        node.trackname = name
        dictionary.update({name: lineagedict})
        if len(node.likelychildren) > 1:
            for i, ch in enumerate(node.likelychildren):
                newname = name+"_"+str(i)
                self.traverseLineage(ch, newname, dictionary)
        else:
            
            return
    def obtainLineages(self):
        roots = [n for n in self.nodes if len(n.likelyparent)==0]
        self.namingdict = {}
        for i,r in enumerate(roots):
            name = "Cell_"+str(i)
            self.traverseLineage(r,name,self.namingdict)

    def getLineageDict(self):
        return self.namingdict

    def virtualizeCoords(self):
        for n in self.nodes:
            n.vcoords = []
        occupiedlist = [[0] for i in range(len(self.images))]
        for n in self.nodes:
            self.traverseTree(n, occupiedlist)

    def linknodes(self, parentcoords, childcoords):
        parent = [n for n in self.nodes if n.name ==
                  parentcoords[1] and n.frame == parentcoords[0]][0]
        child = [n for n in self.nodes if n.name ==
                 childcoords[1] and n.frame == childcoords[0]][0]
        child.likelyparent = [parent]
        if child not in parent.likelychildren:
            parent.likelychildren.append(child)
            
        self.makeTree()
        self.virtualizeCoords()
        self.obtainLineages()
    def filterByConvexity(self,convexity = 0.8):
        removal_list = []
        for n in self.nodes:
            if n.convexity < convexity:
                removal_list.append([n.frame,n.name])
        return removal_list
    
    def filterByAspectRatio(self,ar = 1.1):
        removal_list = []
        for n in self.nodes:
            if n.major/n.minor < ar:
                removal_list.append([n.frame,n.name])
        return removal_list
    def filterByArea(self,area = 100):
        removal_list = []
        for n in self.nodes:
            if n.area < area:
                removal_list.append([n.frame,n.name])
        return removal_list
    
if __name__ == '__main__':
    x, d, t, n, net = main()

    counter = 0
    for node in net.nodes:
        if len(node.likelyparent) > 1:
            counter += 1
