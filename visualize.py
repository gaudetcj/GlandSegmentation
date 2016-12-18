# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 08:32:42 2016

@author: Chase
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from submission import prep
from data import image_cols, image_rows


def visualize_results():
    images  = np.load('imgs_test.npy')
    results = np.load('imgs_mask_test.npy')
    
    total = results.shape[0]
    for i in range(total):
        image = images[i,0]
        image = cv2.resize(image, (image_cols, image_rows))
        result = results[i,0]
        result = prep(result)
        
        plt.imshow(image, cmap='Greys')
        plt.imshow(result, alpha=0.30)
        plt.show()


if __name__ == '__main__':
    visualize_results()