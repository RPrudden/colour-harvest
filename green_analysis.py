from __future__ import division
import sys
import scipy
from scipy import ndimage as ndi
from skimage import measure
from skimage import data
from skimage import filters
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import itertools
import urllib
import time

target_yellow = np.array([255, 202, 80])#np.array([199, 159, 50])
target_green = np.array([2, 123, 3])#([36, 64, 33])
target_blue = np.array([6, 159, 211])#np.array([16, 64, 122])
norm_yellow = target_yellow / target_yellow[0]
norm_green = target_green / target_green[0]
norm_blue = target_blue / target_blue[0]
norms = [norm_yellow, norm_green, norm_blue]
targets = [target_yellow, target_green, target_blue]

def score_colours(feats):
    avgs = [np.array((np.mean(ft[:,:,0]),np.mean(ft[:,:,1]),np.mean(ft[:,:,2]))) for ft in feats]
    norm_colours = [c/c[0] for c in avgs]
    order = list(itertools.permutations([norm_yellow,norm_green,norm_blue]))
    scores = np.array([np.sum(np.array((np.array(norm_colours) - np.array(targets))**2)) for targets in order])
    return (order[np.argmin(scores)], np.min(scores))

def find_colours_grass(feats):
    all_scores = []
    for blocks, i in zip(list(itertools.combinations(feats, 3)), list(itertools.combinations(range(len(feats)), 3))):
        all_scores.append((blocks,score_colours(blocks)[1],i))
    minscore = np.min(np.array(all_scores)[:,1])
    blocks = None
    for b, score, i in all_scores:
        if score == minscore:
            blocks = b
            blocksi = i
    grassi = [a for a in range(len(feats)) if not a in blocksi][0]
    grass = feats[grassi]
    return (blocks, grass)

def get_targets(ntargets):
    result = []
    for t in ntargets[0]:
        for i,n in enumerate(norms):
#             if all(n == t):
            if np.array_equal(n,t):
                result.append(targets[i])
                break
    return result

def get_diffs(feats, ntargets):
    avgs = [np.array((np.mean(ft[:,:,0]),np.mean(ft[:,:,1]),np.mean(ft[:,:,2]))) for ft in feats]
    target_colours = get_targets(ntargets)
    return np.array(avgs) - np.array(target_colours)

def get_total_diffs(feats, ntargets):
    diffs = get_diffs(feats, ntargets)
    return np.array([np.mean(diffs[:,i]) for i in range(3)])

def get_green(img_file):
    img = ndi.imread(img_file)

    im = filters.gaussian_filter(img, sigma=256 / (40. * 20))
    blobs = im < im.mean()
    red_blobs = blobs[:,:,0][:,:,None]
    blue_blobs = blobs[:,:,2][:,:,None]

    features = []
    for blobs in [red_blobs, blue_blobs]:
        all_labels, n = measure.label(blobs, return_num=True)
        for i in range(n):
            blob = np.ma.masked_where(all_labels != i, all_labels)
            m = np.repeat(blob.mask,3,axis=2)
            ft = np.ma.masked_array(img, mask=m)
            ft.fill_value = 255
            avr,avg,avb = np.mean(ft[:,:,0]),np.mean(ft[:,:,1]),np.mean(ft[:,:,2])
            colour = np.array([avr,avg,avb])
            norm_colour = colour / colour[0]
            score = (avr/avg) / (avr/avb)
            score = score if score > 1 else 1/score
            if ft.count() > 3000 and ft.count() < 200000 and score > 1.1:
                features.append((ft, score))
                
    features = sorted(features, key=lambda x: x[1])[:4]
    features = [f[0] for f in features]
    features, grass = find_colours_grass(features)




    total_diffs = get_total_diffs(features, score_colours(features))/255

    grass_col = (np.mean(grass[:,:,0]),np.mean(grass[:,:,1]),np.mean(grass[:,:,2]))
    norm_grass_col = np.floor(grass_col*(1-total_diffs))
    return norm_grass_col

if __name__ == "__main__":
    while True:
        urllib.urlretrieve("http://192.168.2.102/", './image')
        g = [20,200,20]
        try:    
            g = get_green('./image')
        except:
            pass
        with open("data.csv", "w") as f:
            f.write(str(g[0])+','+str(g[1])+','+str(g[2]))
    time.sleep(10)
