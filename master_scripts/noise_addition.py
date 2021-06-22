import numpy as np

###
import scipy.optimize as opt
import math
import sys
sys.path.append("../../../master_scripts")
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import cauchy
from master_scripts.plotting import plot_roc_curve
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import normalize
from master_scripts.analysis_functions import anodedata_classification
from scipy.stats import norm
import pandas as pd
import copy
import pickle
import random
###

#rdfile = np.genfromtxt("../../data/real/ratiodist.txt")




class Noiser:
    
    ### Data description
    # The primary datastructure used in this class is a dictionary of "maps"
    # The maps are three-dimentional lists which contain noise distributions
    # relating to a pixel some distance away from the center of the event.
    # The maps are generated as follows:
    #     A set of experimentally collected images, "real images", are fit to
    #     a two-dimensional Lorentzian curve.
    #     The fractional difference between each pixel in the real image and
    #     the fit is appended to several distributions according to how far
    #     that pixel was away from the center of the fit.
    # The distributions are marshalled and unmarshalled from this map using
    # a dynamically generated intermediate datastructure which maps the pixels
    # in an images to distributions in the noise map, implemented as a pandas
    # dataframe acting as a 2-d array of tuples where the tuples contain
    # the coordinate displacement from the center of the event.
    # 
    # Since creating the dictionary of maps is very expensive,
    # the dictionary is saved as a pickle file to be loaded by default.
    # The file must be declared at instantiation of the class and should
    # be updated with the "saveDict()" method.
    
    ###
       
    def __init__(self,dictFile="dictFile.pickle",ampFile="ampFile.pickle",new=False):
        self.dictFile = dictFile
        self.ampFile = ampFile
        self.mapDict = {}
        self.ampDict = {}
        if new==False:
            self.mapDict = self.load_object(dictFile)
            self.ampDict = self.load_object(ampFile)
        
    def save_object(self,obj,fileName):
        try:
            with open(fileName, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)
        
        
    def load_object(self,filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as ex:
            print("Error during unpickling object (Possibly unsupported):", ex)
    
    
    ### Basic Noise Map Generation Pseudocode
    # Accept ImageSet and Energy of Images
    # Generate MapMat
    # UpdateAll MapMat from ImageSet
    # Add finished MapMat to mapDict with Energy as the key
    ###
    
    def saveDict(self):
        self.save_object(self.mapDict,self.dictFile)
    
    def newMap(self, imageSet, energy):
        mapMat = self.genMapMat()
        self.updateMapAll(mapMat,imageSet)
        self.mapDict[energy] = mapMat
    
    ###
    # I'm condsidering the addition of a new map being ab-initio only
    # That is, we never update a map, we only generate a new one
    # This might help us avoid problems where values are inverted twice
    # or distributions being left unsorted etc.
    # It's an expensive operation, but should only have to be done rarely
    # and is absolutely worth avoiding data-corruption based confusion.
    ###
    
    def genMapMat(self):
        mapMat = []
        for i in range(0,33):
            mapMat.append([])
        for i in range(0,33):
            for j in range(0,33):
                mapMat[i].append([])
        return mapMat
    
    def updateMap(self, mapMat,image):
        try:
                fracDiffMat = np.array(self.diffLor(image)).reshape(16,16)
        except ValueError:
            #print(image)
            return 0
        Y, X = self.locLor(image)
        mapdf = self.mapNumMat(X,Y)

        for i in range(0,16):
            for j in range(0,16):
                (a,b) = mapdf[i][j]
                mX = 16+int(np.round(a))
                mY = 16+int(np.round(b))
                if i != 0 and i != 15 and not(i == 3 and j == 13) and not(i == 7 and j == 11):
                    mapMat[mX][mY].append(
                        fracDiffMat[i][j])
    
    def genAvgMap(self, mapMat):
        avgMap = np.zeros((33,33),dtype=float)
        for i in range(0,33):
            for j in range(0,33):
                avgMap[i][j] = np.average(mapMat[i][j])
        return avgMap
    
    ## BAD Updates in place
    def updateMapAll(self, imgMap,imgSet):
        for i in range(0,len(imgSet)):
            try:
                self.updateMap(imgMap,imgSet[i])
            except:
                print("Unable to update map from image: ",i)
        #imgMap = matInv(imgMap)
        imgMap = self.matSort(imgMap)
    
    def matInv(self, mat):
        invMat = self.genMapMat()
        for i in range(0,len(mat[0])):
            for j in range(0,len(mat[0])):
                invMat[i][j] = list(map(lambda x: 1/x, mat[i][j]))
        return invMat
    
    def matSort(self, mat):
        sortMat = self.genMapMat()
        for i in range(0,len(mat[0])):
            for j in range(0,len(mat[0])):
                sortMat[i][j] = sorted(mat[i][j])
        return sortMat
    
    ### Depricated function
    #def applyMap(mapMat,image):
    #    nimage = copy.deepcopy(image)
    #    Y, X = locLor(image)
    #    mapdf = mapNumMat(X,Y)

    #    for i in range(0,16):
    #        for j in range(0,16):
    #            (a,b) = mapdf[i][j]
    #            mX = 16+int(np.round(a))
    #            mY = 16+int(np.round(b))
    #            if i != 0 and i != 15 and not(i == 3 and j == 13) and not(i == 7 and j == 11):
    #                randomPick = np.random.choice(mapMat[mX][mY])
    #                nimage[i][j] = nimage[i][j]*(1/randomPick)
    #    return nimage
    
    def dist_frac(self, dist,pick):
        return dist.index(pick)/len(dist)
    
    def val_from_frac(self, dist, frac):
        index = int(np.round(frac * len(dist)))
        return dist[index]
    
    
    def applyMap(self, mapMat,image):
        nimage = copy.deepcopy(image)
        Y, X = self.locLor(image)
        mapdf = self.mapNumMat(X,Y)

        scalingPick = self.dist_frac(mapMat[16][16],np.random.choice(mapMat[16][16]))
        for i in range(0,16):
            for j in range(0,16):
                (a,b) = mapdf[i][j]
                mX = 16+int(np.round(a))
                mY = 16+int(np.round(b))
                if i != 0 and i != 15 and not(i == 3 and j == 13) and not(i == 7 and j == 11):
                    nimage[i][j] = nimage[i][j]*1/(self.val_from_frac(mapMat[mX][mY],scalingPick))
                else:
                    nimage[i][j] = 0
        return nimage
    
    def getNoise(self, mapMat,image):
        nimage = copy.deepcopy(image)
        Y, X = locLor(image)
        mapdf = mapNumMat(X,Y)
        noiseMat = np.zeros((16,16),dtype=float)

        scalingPick = dist_frac(mapMat[16][16],np.random.choice(mapMat[16][16]))
        for i in range(0,16):
            for j in range(0,16):
                (a,b) = mapdf[i][j]
                mX = 16+int(np.round(a))
                mY = 16+int(np.round(b))
                if i != 0 and i != 15 and not(i == 3 and j == 13) and not(i == 7 and j == 11):
                    noiseMat[i][j] = 1/(val_from_frac(mapMat[mX][mY],scalingPick))
        return noiseMat

    
    def diffLor(self,image, bg=False, blur=False, sigma=1):
        Xin, Yin = np.mgrid[0:16, 0:16]
        initial_guess = (3, 5, 5, 2, 2.2)
        try:
            popt, pcov = opt.curve_fit(self.twoD_Lor, (Xin, Yin) , image.flatten(), p0=initial_guess)
        except RuntimeError:
            return 0
        #(height, Yin, Xin, width_y, width_x) = popt

        genImage = self.genLor(*popt)
        if blur:
            genImage = gaussian_filter(genImage, sigma=sigma)
        genImage[0] = [0 for x in genImage[0]]
        genImage[-1] = [0 for x in genImage[-1]]
        genImage[3][13] = 0
        genImage[7][11] = 0

        if bg:
            genImage[np.where((genImage <= 500) & (genImage != 0))] = 500

        genImg = genImage.flatten()
        img = image.flatten()
        retImg = []

        for i in range(0, len(img)):
            if img[i] == 0 and genImg[0] == 0:
                retImg.append(1)
                #retImg.append(genImg[i])
            else:
                retImg.append(genImg[i] / img[i])

        return retImg

    def locLor(self,image):
        Xin, Yin = np.mgrid[0:16, 0:16]
        initial_guess = (3, 5, 5, 2, 2.2)
        try:
            popt, pcov = opt.curve_fit(self.twoD_Lor, (Xin, Yin) , image.flatten(), p0=initial_guess)
        except RuntimeError:
            return 0
        (height, Yin, Xin, width_y, width_x) = popt

        return (Yin,Xin)

    
    def twoD_Lor(self,xdata_tuple, amp,center_x, center_y, width_x, width_y):
        """Returns a lorentzian function with the given parameters"""
        (x,y) = xdata_tuple
        width_x = float(width_x)
        width_y = float(width_y)
        return ((amp*((.5*width_x)/((x-center_x)**2 + (.5*width_x)**2))) 
                * (((.5*width_y)/((y-center_y)**2 + (.5*width_y)**2)))).ravel()
    
    
    def genLor(self, amp,x,y,width_x,width_y):
        Xin, Yin = np.mgrid[0:16, 0:16]
        img = self.twoD_Lor((Xin, Yin), amp, x, y, width_x, width_y)
        return img.reshape(16,16)


    def mapNumMat(self, x,y):
        df = pd.DataFrame([])
        for i in range(0,16):
            mapList = []
            for j in range(0,16):
                a = i - x
                b = j - y
                mapList.append((a,b))
            df = df.append(pd.Series(mapList),ignore_index=True)
        return df
    
    def paramLor(self,data):
        Xin, Yin = np.mgrid[0:16, 0:16]
        initial_guess = (3, 5, 5, 2, 2.2)
        try:
            popt, pcov = opt.curve_fit(self.twoD_Lor, (Xin, Yin) , data.flatten(), p0=initial_guess)
        except:
            return [-1,-1,-1,-1,-1]
        return popt
    
    def saveAmp(self):
        self.save_object(self.ampDict,self.ampFile)
    
    
    def newAmp(self, choices, energy):
        ampDist = self.ampDists(choices)
        self.ampDict[energy] = ampDist
    
    def ampDists(self,choices):
        choiceParams = []
        for i in range(0,len(choices)):
            tempParams = self.paramLor(choices[i])
            
            if tempParams[0] == -1 and tempParams[1] == -1 and tempParams[2] == -1 and tempParams[3] == -1 and tempParams[4] == -1:
                continue
                
            choiceParams.append(self.paramLor(choices[i]))

        choiceAmps = [x[0] for x in choiceParams]    

        return np.array(choiceAmps)
    
    
    def genImage(self,maxVal,ex,ey):
        x = maxVal
        ax = -4.12594601e-07
        bx = 2.37017912e-03
        cx = 2.54213157e+00
        resx = ax*x**2+bx*x+cx
        #print("X width: ", resx)

        ay = -4.08014131e-07
        by = 2.27804971e-03  
        cy = 2.26606974e+00
        resy = ay*x**2+by*x+cy
        #print("Y width: ", resy)

        Xin, Yin = np.mgrid[0:16, 0:16]
        #amps = ampDists(chosen(rimages[decay], maxVal))
        mu, std = norm.fit(self.ampDict[maxVal])
        #print(self.ampDict[maxVal])
        #print(mu)
        img = self.twoD_Lor((Xin, Yin), mu, ex, ey, resx, resy)
        return img.reshape(16,16)
    
    def genRandImage(self,dummy):
        #noiser.applyMap(noiser.mapDict[3500],noiser.genImage(3500,12,12).reshape(16,16))
        ##DEBUG
        double = 0
        
        image = np.zeros((16,16),dtype=float)
        double = np.random.randint(1,3)
        #if double == 1
        for i in range(0,double):
            eRange = list(self.mapDict.keys())
            yRange = range(1,16)
            energy = random.choice(eRange)
            if i == 0:
                y1 = random.choice(yRange)
                x1 = random.choice(range(0,16))
            else:
                y2 = random.choice(yRange)
                x2 = random.choice(range(0,16))
            try:
                if i == 0:
                    image += self.applyMap(self.mapDict[energy],self.genImage(energy,x1,y1))
                else:
                    image += self.applyMap(self.mapDict[energy],self.genImage(energy,x2,y2))
            except:
                #print("Index error, trying again")
                #print("DEBUG: ",double)
                return self.genRandImage()
        if double == 2:
            return [image,double-1,[[x1,y1],[x2,y2]],energy]
        if double == 1:
            return [image,double-1,[x1,y1],energy]
        else:
            print("WHAT",i,double)
            return [image,double-1,[x1,y1]]
        
        
    
    
    
    
###
# The following code is legacy and depricated, only kept in comment form to
# help understanding of code and data relating to it.
# Eventually it will be removed entirely when no surviving code relates to it.
###


#def gen_dist(file_path):
#    #Given a path, build a 2d array
#    rdfile = np.genfromtxt(file_path)
#    dist = []
#    x = rdfile[:,0]
#    #The second columns needs to be retyped to satisfy range()
#    y = list(map(int,rdfile[:,1]))
#    #This iterates over each bin (1st col) and appends it's value
#    #to the list per value in the 2nd col
#    #Some bins added many times others not at all if y[i] == 0
#    #This will create the full distribution, ready to be randomly
#    #sampled from
#    for i in range(len(x)):
#        for j in range(y[i]):
#            dist.append(x[i])
#    return np.asarray(dist)

#def rnoise_gen(dist):
##    dist = np.asarray(gen_dist(file_path))
#    #This generates a flat array of numbers randomly sampled from the distribution
#    random_noise = np.asarray([np.random.choice(dist) for i in range(0,16**2)])
#    return random_noise
    
    
    
