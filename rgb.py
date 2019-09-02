from __future__ import print_function
import zipfile
import os
#from sys import platform
import shutil
#IREM
import cntk as C
from cntk.initializer import glorot_uniform
from cntk.ops import relu, sigmoid, input_variable
from cntk.ops.functions import load_model
import random
from collections import defaultdict
import csv
import sys
import cv2
import xlsxwriter
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pylab import *
from tqdm import tqdm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.util import montage2d
from skimage.morphology import label

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense, MaxPooling
from cntk.ops import element_times, relu, sigmoid, combine, softmax, as_composite
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk import Trainer, load_model, Axis, input_variable, parameter, times, combine, softmax, roipooling, plus, element_times, CloneMethod, alias, Communicator, reduce_sum
from cntk.layers import placeholder, Constant, Sequential, ConvolutionTranspose

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

#csv.field_size_limit(sys.maxsize)
maxInt = sys.maxsize
decrement = True

while decrement:
	# decrease the maxInt value by factor 10 
	# as long as the OverflowError occurs.

	decrement = False
	try:
		csv.field_size_limit(maxInt)
	except OverflowError:
		maxInt = int(maxInt/10)
		decrement = True

bands=['A', 'M', 'P']
# Load grid size
x_max=[]
y_min=[]
IM_ID=[]
for _im_id, _x, _y in csv.reader(open('D:/KaggleDataset/dstl-satellite-imagery-feature-detection/grid_sizes2.csv')):
	x_max.append(float(_x))
	y_min.append(float(_y))
	IM_ID.append(_im_id)
	
# Load train poly with shapely
train_polygons=[]
POLY_TYPE=[]
IM_ID_TYPE=[]
csvreader=csv.reader(open('D:/KaggleDataset/dstl-satellite-imagery-feature-detection/train_wkt_v4.csv','r'),quotechar='"',delimiter=',',quoting=csv.QUOTE_ALL,skipinitialspace=True)
next(csvreader) # skip the heading
for _im_id, _poly_type, _poly in csvreader:
    IM_ID_TYPE.append(_im_id)
    train_polygons.append(shapely.wkt.loads(_poly))
    POLY_TYPE.append(_poly_type) 	
	
trained=[]
n=0
for listitem in range(0,25):
    trained.append(IM_ID_TYPE[n*10])
    n=n+1

#Read image with tiff
im_rgb=[]
im_size_rgb=[]
index=0
# Use IM_ID_TYPE rather than IM_ID if you need only the images in the training set, not all of them!!
for id in IM_ID_TYPE:
	im_rgb.append(tiff.imread('D:/KaggleDataset/dstl-satellite-imagery-feature-detection/three_band/three_band/{}.tif'.format(id)).transpose([1,2,0]))
	im_size_rgb.append(im_rgb[index].shape[:2])
	index+=1
	
#START	
h_=[]
w_=[]
h_s=[]
w_s=[]
for h_w in im_size_rgb:
    h_.append(h_w[0]*(h_w[0]/(h_w[0]+1)))
    w_.append(h_w[1]*(h_w[1]/(h_w[1]+1)))
    h_s.append(h_[0]/y_min[0])
    w_s.append(w_[0]/x_max[0])

train_polygons_scaled=[]	
s=0
for ind in train_polygons:
    train_polygons_scaled.append(shapely.affinity.scale(ind, xfact=w_s[s], yfact=h_s[s], origin=(0, 0, 0)))
    s+=1	

#obtain the images and the corresponding sizes of the trained images ONLY (total of 25 images with 10 batches for 10 classes)
only_trained_rgb=[]
only_trained_size_rgb=[]
index=0
for id in IM_ID_TYPE:
    only_trained_rgb.append(tiff.imread('D:/KaggleDataset/dstl-satellite-imagery-feature-detection/three_band/three_band/{}.tif'.format(id)).transpose([1,2,0]))
    only_trained_size_rgb.append(only_trained_rgb[index].shape[:2])
    index+=1	

#IREM
train_mask=[]
k=0
for listitem in train_polygons_scaled:
    img_mask=np.zeros(only_trained_size_rgb[k], np.uint8)

    int_coords=lambda x: np.array(x).round().astype(np.int32)
    exteriors=[int_coords(poly.exterior.coords) for poly in listitem]
    interiors=[int_coords(pi.coords) for poly in listitem for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    train_mask.append(img_mask)
    k+=1

	
input_images=[]
target_masks=[]

# 1 - For 6040_2_2
ind=0
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])-1
maskk=[]
inx=[]
for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)


# 2 - For 6120_2_2
ind=1
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+9
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 3 - For 6120_2_0
ind=2
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+19
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 4 - For 6090_2_0
ind=3
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+29
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 5 - For 6040_1_3
ind=4
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+39
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 6 - For 6040_1_0
ind=5
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+49
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 7 - For 6100_1_3
ind=6
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+59
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 8 - For 6010_4_2
ind=7
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+69
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 9 - For 6110_4_0
ind=8
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+79
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 10 - For 6140_3_1
ind=9
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+89
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 11 - For 6110_1_2
ind=10
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+99
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 12 - For 6100_2_3
ind=11
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+109
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 13 - For 6150_2_3
ind=12
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+119
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 14 - For 6160_2_1
ind=13
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+129
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 15 - For 6140_1_2
ind=14
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+139
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 16 - For 6110_3_1
ind=15
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+149
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 17 - For 6010_4_4
ind=16
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+159
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 18 - For 6170_2_4
ind=17
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+169
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 19 - For 6170_4_1
ind=18
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+179
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 20 - For 6170_0_4
ind=19
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+189
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 21 - For 6060_2_3
ind=20
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+199
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)

# ******22 - For 6070_2_3
# ind=21
# indd=ind*10
# k=0
# ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
# Class=[5]
# img_0=only_trained_rgb[indd]
# a=(Class[0])+209
# maskk=[]
# inx=[]

# for i in ss:
    # for j in ss:
        # k+=1
        # inn=[]
        # img_1=img_0[i:(i+224), j:(j+224)]
        # img_2=img_1.copy()
        # new_mask=train_mask[a]
        # new_mask=new_mask[i:(i+224), j:(j+224)]
        # new_mask = np.asarray(new_mask)
        # channel1=img_1[: ,: ,0]
        # inn.append(channel1)
        # channel2=img_1[: ,: ,1]
        # inn.append(channel2)
        # channel3=img_1[: ,: ,2]
        # inn.append(channel3)
        # inn=np.asarray(inn)
        # inx.append(inn)	
        # maskk.append(new_mask)
# target_masks.append(maskk)
# input_images.append(inx)
		
# 22 - For 6010_1_2
ind=22
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+219
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 23 - For 6040_4_4
ind=23
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+229
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
		
# 24 - For 6100_2_2
ind=24
indd=ind*10
k=0
ss=[0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688]
Class=[5]
img_0=only_trained_rgb[indd]
a=(Class[0])+239
maskk=[]
inx=[]

for i in ss:
    for j in ss:
        k+=1
        inn=[]
        img_1=img_0[i:(i+224), j:(j+224)]
        img_2=img_1.copy()
        new_mask=train_mask[a]
        new_mask=new_mask[i:(i+224), j:(j+224)]
        new_mask = np.asarray(new_mask)
        channel1=img_1[: ,: ,0]
        inn.append(channel1)
        channel2=img_1[: ,: ,1]
        inn.append(channel2)
        channel3=img_1[: ,: ,2]
        inn.append(channel3)
        inn=np.asarray(inn)
        inx.append(inn)	
        maskk.append(new_mask)
target_masks.append(maskk)
input_images.append(inx)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

input_images = np.asarray(input_images, dtype=np.float32)
target_masks = np.asarray(target_masks, dtype=np.float32)

print('first input:',input_images[0])
print('first target:', target_masks.shape, 'first input:', input_images.shape)

print(target_masks[0])
input_images = np.reshape(input_images, (4056, 3, 224, 224)) 
target_masks = np.reshape(target_masks, (4056, 1, 224, 224))
print(input_images.shape)
print(input_images[0].shape)
print(input_images[0])
print(target_masks.shape)
print(target_masks[0])
# biy=input_images[0]
# iyy=np.reshape(biy, (height, width, channel)) 
# imshow(255*iyy.astype(np.uint8))
# plt.show()

# workbook=xlsxwriter.Workbook('secondim.xlsx')
# worksheet=workbook.add_worksheet()
# row=0
# col=0
# for a in input_images[0][0]:
    # worksheet.write(row,col,str(a))
    # row+=1
# workbook.close()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def UpSampling2D(x):
    xr = C.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
    print(xr.shape)
    xx = C.splice(xr, xr, axis=-1) # axis=-1 refers to the last axis
    xy = C.splice(xx, xx, axis=-3) # axis=-3 refers to the middle axis
    print(xy.shape)
    r = C.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))
    print(r.shape)
    return r

def create_model(input, num_classes):
    print(input.shape)
    conv1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True, name = "bir")(input)
    conv1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True, name = "iki")(conv1)
    print(conv1.shape)
    pool1 = MaxPooling((2,2), strides=(2,2))(conv1)
    
    conv2 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(pool1)
    conv2 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(conv2)
    print(conv2.shape)
    pool2 = MaxPooling((2,2), strides=(2,2))(conv2)

    conv3 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(pool2)
    conv3 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(conv3)
    print(conv3.shape)
    pool3 = MaxPooling((2,2), strides=(2,2))(conv3)

    conv4 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(pool3)
    conv4 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(conv4)
    print(conv4.shape)
    pool4 = MaxPooling((2,2), strides=(2,2))(conv4)
    
    conv5 = Convolution((3,3), 512, init=glorot_uniform(), activation=relu, pad=True)(pool4)
    conv5 = Convolution((3,3), 512, init=glorot_uniform(), activation=relu, pad=True)(conv5)
    print(conv5.shape)
    up6 = C.splice(UpSampling2D(conv5), conv4, axis=0)
    conv6 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(up6)
    conv6 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(conv6)
    print(conv6.shape)
    up7 = C.splice(UpSampling2D(conv6), conv3, axis=0)
    conv7 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(up7)
    conv7 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(conv7)
    print(conv7.shape)
    up8 = C.splice(UpSampling2D(conv7), conv2, axis=0)
    conv8 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(up8)
    conv8 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(conv8)
    print(conv8.shape)
    up9 = C.splice(UpSampling2D(conv8), conv1, axis=0)
    conv9 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(up9)
    conv9 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True, name = "beforelast")(conv9)
    print(conv9.shape)
    conv10 = Convolution((1,1), num_classes, init=glorot_uniform(), activation=sigmoid, pad=True, name = "last")(conv9)
    print(conv10.shape)
    return conv10

def dice_coefficient(x, y):
    # average of per-channel dice coefficient

    intersection = C.reduce_sum(x * y, axis=(1,2))

    return C.reduce_mean(2.0 * intersection / (C.reduce_sum(x, axis=(1,2)) + C.reduce_sum(y, axis=(1,2)) + 1.0))
	
def slice_minibatch(data_x, data_y, i, minibatch_size):
    sx = data_x[i * minibatch_size:(i + 1) * minibatch_size]
    sy = data_y[i * minibatch_size:(i + 1) * minibatch_size]
    return sx, sy

def measure_error(data_x, data_y, x, y, trainer, minibatch_size):
    errors = []
    print('trainings:',data_x.shape,data_y.shape,x.shape,y.shape,minibatch_size)
    for i in range(0, int(len(data_x) / minibatch_size)):
        data_sx, data_sy = slice_minibatch(data_x, data_y, i, minibatch_size)
        errors.append(trainer.test_minibatch({x: data_sx, y: data_sy}))

    return np.mean(errors)

def train(images, masks, use_existing=False):
    shape = input_images[0].shape
    print('all',input_images[0].shape, input_images.shape[0],masks[0].shape)
    data_size = input_images.shape[0]

    # Split data
    test_portion = int(data_size * 0.1)
    indices = np.random.permutation(data_size)
    test_indices = indices[:test_portion]
    training_indices = indices[test_portion:]

    test_data = (images[test_indices], masks[test_indices])
    training_data = (images[training_indices], masks[training_indices])

    # Create model
    x = C.input_variable(shape)
    y = C.input_variable(masks[0].shape)
    print(len(training_data[0]),x.shape,y.shape,masks.shape[1])
    z = create_model(x, masks.shape[1])
    dice_coef = dice_coefficient(z, y)
    # Load the saved model if specified
    checkpoint_file = "cntk-unet.dnn"
    if use_existing:
        z.load_model(checkpoint_file)
	
    # Prepare model and trainer
    lr = C.learning_rate_schedule(0.000001, C.UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0.9)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))


    # Get minibatches of training data and perform model training
    minibatch_size = 8
    num_epochs = 50

    training_errors = []
    test_errors = []

    for e in range(0, num_epochs):
        for i in range(0, int(len(training_data[0]) / minibatch_size)):
            data_x, data_y = slice_minibatch(training_data[0], training_data[1], i, minibatch_size)

            trainer.train_minibatch({x: data_x, y: data_y})

        # Measure training error
        training_error = measure_error(training_data[0], training_data[1], x, y, trainer, minibatch_size)
        training_errors.append(training_error)

        # Measure test error
        test_error = measure_error(test_data[0], test_data[1], x, y, trainer, minibatch_size)
        test_errors.append(test_error)

        print("epoch #{}: training_error={}, test_error={}".format(e, training_errors[-1], test_errors[-1]))

        checkpoint_file = "irem.dnn"
        trainer.save_checkpoint(checkpoint_file) 
    return trainer, training_errors, test_errors

trainer, training_errors, test_errors = train(input_images, target_masks)

datasize=input_images.shape[0]
test_p = int(datasize * 0.08)
indi = np.random.permutation(datasize)
test_indi = indi[:test_p]
testd = input_images[test_indi]

pred=trainer.model.eval(testd)
 

tiz=test_indi[0]
img_irem=input_images[tiz]
new_mask_irem=target_masks[tiz]

print('mask shape',new_mask_irem.shape)
print('image shape',img_irem.shape)

pro=np.select([pred[0]<=0.9, pred[0]>0.9], [np.zeros_like(pred[0]), np.ones_like(pred[0])])
# pro= pred[0].astype(int)
preirem=np.squeeze(pro, axis=0)
print('pred shape',preirem.shape)
# new_mask_irem=np.reshape(new_mask_irem, (224,224))
new_mask_irem=np.squeeze(new_mask_irem, axis=0)
print(preirem)
print(new_mask_irem)
fig, (ax1, ax2, ax3)=plt.subplots(1, 3, figsize=(15,5))
ax1.imshow(255*img_irem[0].astype(np.uint8))
ax2.imshow(255*new_mask_irem, cmap='gray')
ax3.imshow(255*preirem, cmap='gray')
plt.show()