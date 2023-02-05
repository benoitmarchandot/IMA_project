#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 18:32:15 2021

@author: benoitmarchandot
"""

####  Import packages   ####
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from time import time
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
from skimage.feature import corner_fast
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from skimage.measure import find_contours
from sklearn.metrics import matthews_corrcoef, make_scorer
import skimage.morphology as morpho

def download_images(Working_directory, dataset='Train'):
    """This function download all the images of the chosen data_set and returns three lists of all images
    
    input : 
        dataset : Train
                  Test
    
    output :
    
        ori : list of original images of the cells
        segCyt : list of segCyt images
        segNuc : list of sgNuc images
    """
    if dataset not in ['Train','Test'] :
        raise ValueError('dataset must be equal to \'Train\' or \'Test\'')
        
    L = os.listdir('%s/%s/%s'%(Working_directory,dataset,dataset))
    L2 = []
    for x in L:
        if '_' not in x:
            a, b = x.split('.')
            L2.append(int(a))
            
    L2 = sorted(L2)
    ori = []
    segCyt = []
    segNuc = []
    ID = []
    
    for i in range(len(L2)):
        im1, im2, im3 = imread('%s/%s/%s/%s'%(Working_directory,dataset, dataset, L2[i])+'.bmp'), imread('%s/%s/%s/%s'%(Working_directory,dataset, dataset, L2[i])+'_segCyt.bmp'), imread('%s/%s/%s/%s'%(Working_directory,dataset, dataset, L2[i])+'_segNuc.bmp')
        ori.append(im1)
        segCyt.append(im2)
        segNuc.append(im3)
        ID.append(L2[i])
    
    return ori, segCyt, segNuc, ID




def average_intensity(ori, region):
    """This function extracts the nuclear average intensity of every image 
    input : ori, region = segNuc or segCyt 
    
    output : intensity_R, intensity_G, intensity_B the set of average intensity on every channel """
    
    #if region != segCyt and region != segNuc:
        #raise ValueError('region must be equal to segNuc or segCyt')
    
    ori = np.array(ori)
        
    intensity_R = []
    intensity_G = []
    intensity_B = []
    
    for i in range(len(ori)) :
        im = ori[i]
        R, G, B = im[:,:,0], im[:,:,1], im[:,:,2]
        equ_R, equ_G, equ_B = cv.equalizeHist(R), cv.equalizeHist(G), cv.equalizeHist(B)
        
        seg = region[i]
        intensity_R.append(np.mean(equ_R[seg>0],axis=0))
        intensity_G.append(np.mean(equ_G[seg>0],axis=0))
        intensity_B.append(np.mean(equ_B[seg>0],axis=0))
        
    return intensity_R, intensity_G, intensity_B


def average_contrast(ori, region):
    
    """This function gives the features of the image which corresponds to the average contrast of the region
    input : ori (original image), region (segNuc or segCyt)
    ouput : contrast_R, contrast_G, contrast_B the set of average contrast on every channel"""
    
    ori = np.array(ori)    #computing the gradient norm of the image
    se=morpho.selem.disk(1)
        
    contrast_R = []
    contrast_G = []
    contrast_B = []
    
    for i in range(len(ori)) :
        im = ori[i]
        grad_R = morpho.dilation(im[:,:,0], se)- morpho.erosion(im[:,:,0], se)
        grad_G = morpho.dilation(im[:,:,1], se)- morpho.erosion(im[:,:,1], se)
        grad_B = morpho.dilation(im[:,:,2], se)- morpho.erosion(im[:,:,2], se)
        seg = region[i]
        contrast_R.append(np.mean(grad_R[seg>0],axis=0))
        contrast_G.append(np.mean(grad_G[seg>0],axis=0))
        contrast_B.append(np.mean(grad_B[seg>0],axis=0))
        
    return contrast_R, contrast_G, contrast_B



def area(region):
    """This function returns the feature area of the region of the nuc or of the cyt
    input : region (segNuc or segCyt)
    output : area"""
    region = np.array(region)
        
    area = []
    
    for i in range(len(region)) :
        seg = region[i]
        n = len(seg[seg>0])/(seg.shape[0]*seg.shape[1])
        area.append(n)
    return area
    

def corner_detection(ori, region):
    '''This function extracts shapes features of the chosen region of the images using the Shi-Tomasi corner detection from opencv library
    input  : region (segNuc or segCyt)
    output : n_points  (the number of points of corner in each image)
             dist_min_corner  (the minimum distance between the points)
             dist_max_corner (the maximum distance between the points)
             centers (the centers of the corner zone)
             '''
    region = np.array(region)
    ori = np.array(ori)
    n_points = []
    dist_min_corner = []
    dist_max_corner = []
    x_center = []
    y_center = []
    for i in range(len(region)):    #compute the corners
        seg, im = region[i], ori[i]
        img = np.zeros_like(im)
        img[:,:,0] = seg
        img[:,:,1] = seg
        img[:,:,2] = seg
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        
        if not isinstance(corners, (list, tuple, set, np.ndarray)):
            n_points.append(0)
            dist_min_corner.append(0)
            dist_max_corner.append(0)
            x_center.append(0)
            y_center.append(0)
        else :
            corners = np.int0(corners)
            for i in corners:
                x,y = i.ravel()
                cv2.circle(img,(x,y),3,255,-1)
            corners = corners.tolist()
            
            for i in range(len(corners)):
                corners[i] = corners[i][0]
            corners = 1/(im.shape[0]*im.shape[1])*np.array(corners, dtype=float)

            kmeans = KMeans(init='random', n_clusters=1, max_iter=100, n_init=100).fit(corners)
            center = kmeans.cluster_centers_
            x_center.append(center[0,0])
            y_center.append(center[0,1])
            n_points.append(len(corners))
        
            dist = [] #compute the distance between corner points
            for i in range(len(corners)):
                for j in range(len(corners)):
                    if i!=j:
                        dist.append(((corners[i,0]-corners[j,0])**2+ (corners[i,1]-corners[j,1])**2)**0.5)
                    
            dist_min_corner.append(min(dist))
            dist_max_corner.append(max(dist))
    return n_points, dist_min_corner, dist_max_corner, x_center, y_center



def ratio_NC(segCyt, segNuc):
    """This function returns the ratio between the cytoplasm and the nucleous
    input : both regions
    output : list of the ratios"""
    segCyt, segNuc = np.array(segCyt), np.array(segNuc)
    ratio = []
    
    for i in range(len(segNuc)) :
        nuc = segNuc[i]
        cyt = segCyt[i]
        n_nuc = len(nuc[nuc>0])
        n_cyt = len(cyt[cyt>0])
        ratio.append(n_nuc/n_cyt)
    return ratio



def longest_diameter(segCyt, segNuc):
    """This function returns the shortest diameter of the contour of the choosen region
    input : segCyt, segNuc
    output : list of the shortest diameter of the choosen region"""
    
    diameters_cyt = []
    diameters_nuc = []
    for i in range(len(segCyt)):
        cyt, nuc = segCyt[i], segNuc[i]
        contours_nuc,hierarchy_nuc = cv2.findContours(im_nuc, 1, 2)
        contours_cyt,hierarchy_cyt = cv2.findContours(im_cyt-im_nuc, 1, 2)
        cnt_nuc, cnt_cyt = contours_nuc[0], contours_cyt[0]
        (x_nuc,y_nuc),radius_nuc = cv2.minEnclosingCircle(cnt_nuc)
        (x_cyt,y_cyt),radius_cyt = cv2.minEnclosingCircle(cnt_cyt)
        diameters_cyt.append(int(radius_cyt)/cyt.shape[0]*cyt.shape[1])
        diameters_nuc.append(int(radius_nuc)/nuc.shape[0]*nuc.shape[1])
    return diameters_cyt, diameters_nuc




def checking_convexity(segNuc):
    """This function checks if the nucleous contour is convex or not
    input : list of segmented nucleous
    output : 0 (not convex) or 1 (convex)"""
    convex = []
    for i in range(len(segCyt)):
        nuc = segNuc[i]
        contours_nuc,hierarchy_nuc = cv2.findContours(im_nuc, 1, 2)
        cnt_nuc= contours_nuc[0]
        k = cv2.isContourConvex(cnt_nuc)
        convex.append(int(k))
        
    return convex



def add_features(df, features, sortbyID=False):
    
    if sortbyID :
        df = df.sort_values(by=['ID'])
        
    for i in range(len(features)):
        df['f%s'%i] = features[i]
        print(i)
    return df


