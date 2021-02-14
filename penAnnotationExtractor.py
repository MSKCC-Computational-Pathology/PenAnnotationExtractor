# penAnnotationExtractor.py
# A tool to extract pen annotated regions from WSI thumbnails.
# Created by PJ Schueffler and DVK Yarlagadda, MSKCC
# schueffp@mskcc.org
#
# Schüffler PJ, Yarlagadda DVK, Vanderbilt C, Fuchs TJ: Overcoming an Annotation Hurdle: Digitizing Pen Annotations from Whole Slide Images. Journal of Pathology Informatics, 2021

from pdb import set_trace
import argparse
import sys
from tqdm import tqdm
import pandas as pd
import os
import shutil
import numpy as np
import subprocess
import glob
import time
import cv2
import math
import torch
import skimage.morphology
import concurrent.futures


def filter_small_components(img, min_size):
    labeledc, numc = skimage.morphology.label(img, return_num=True)
    component_sizes = np.unique(labeledc, return_counts=True)
    sel_components = component_sizes[0][component_sizes[1] >= min_size]
    img_filtered = np.zeros(img.shape[:2], dtype=np.uint8)
    img = img.astype(np.uint8)
    for i in sel_components:
        comp_indices = (labeledc==i)
        img_filtered[comp_indices] = img[comp_indices]
    return img_filtered

def extract_molecular_annotation(thumbnail_file, base_dir, iteration):
    
    dilation_radius = (iteration + 1) * 5
    min_component_size = 3000

    slide_id = os.path.splitext(os.path.basename(thumbnail_file))[0]

    img_orig = cv2.imread(thumbnail_file)
    cv2.imwrite(base_dir + '/intermediate/{}_step01.jpg'.format(slide_id), img_orig)

    img = cv2.GaussianBlur(img_orig, (3,3), 0)
    cv2.imwrite(base_dir + '/intermediate/{}_step02.jpg'.format(slide_id), img)

    # Extract tissue over HSV
    hsv_origimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(base_dir + '/intermediate/{}_step03.jpg'.format(slide_id), hsv_origimg)
    tissue_hsv = cv2.inRange(hsv_origimg, np.array([135, 10, 30]), np.array([170, 255, 255]))
    cv2.imwrite(base_dir + '/intermediate/{}_step04.jpg'.format(slide_id), tissue_hsv)
    mask_tissue = tissue_hsv

    # Extract marker
    black_marker = cv2.inRange(hsv_origimg, np.array([0, 0, 0]), np.array([180, 255, 125])) # black marker
    cv2.imwrite(base_dir + '/intermediate/{}_step06.jpg'.format(slide_id), black_marker)

    blue_marker = cv2.inRange(hsv_origimg, np.array([100, 125, 30]), np.array([130, 255, 255])) # blue marker
    cv2.imwrite(base_dir + '/intermediate/{}_step07.jpg'.format(slide_id), blue_marker)

    green_marker = cv2.inRange(hsv_origimg, np.array([40, 125, 30]), np.array([70, 255, 255])) # green marker
    cv2.imwrite(base_dir + '/intermediate/{}_step08.jpg'.format(slide_id), green_marker)

    mask_hsv = cv2.bitwise_or(cv2.bitwise_or(black_marker, blue_marker), green_marker)
    cv2.imwrite(base_dir + '/intermediate/{}_step09.jpg'.format(slide_id), mask_hsv)

    # close gaps in annotations
    dilated_mask = cv2.dilate(mask_hsv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius,dilation_radius)))
    cv2.imwrite(base_dir + '/intermediate/{}_step10.jpg'.format(slide_id), dilated_mask)

    # marker_contours = cv2.findContours(dilated_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    marker_contours = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_enclosed_mask = np.zeros(dilated_mask.shape)
    #cv2.drawContours(marker_enclosed_mask, marker_contours[0], -1, (255,255,255), thickness=-1)
    for cidx in range(len(marker_contours[0])):
        cv2.drawContours(marker_enclosed_mask, marker_contours[0], cidx, (255,255,255), thickness=-1)
    cv2.imwrite(base_dir + '/intermediate/{}_step11.jpg'.format(slide_id), marker_enclosed_mask)

    # filter noise
    mask_filtered = filter_small_components(marker_enclosed_mask, min_component_size)
    cv2.imwrite(base_dir + '/intermediate/{}_step12.jpg'.format(slide_id), mask_filtered)

    # Remove marker from final tissue mask
    mask_filtered = cv2.subtract(mask_filtered, dilated_mask)
    cv2.imwrite(base_dir + '/intermediate/{}_step13.jpg'.format(slide_id), mask_filtered)
    
    # Get tissue enclosed in marker
    tissue_enclosed_marker_mask = cv2.bitwise_and(mask_tissue, mask_filtered)
    cv2.imwrite(base_dir + '/intermediate/{}_step14.jpg'.format(slide_id), tissue_enclosed_marker_mask)
    
    # filter noise
    mask_filtered = filter_small_components(tissue_enclosed_marker_mask, min_component_size)
    tissue_enclosed_marker_mask = mask_filtered.copy()

    # no marker detected
    if np.count_nonzero(tissue_enclosed_marker_mask) < min_component_size:
        cv2.imwrite('data/thumbnails/' + str(iteration + 1) + '/{}.jpg'.format(slide_id), img_orig)
    else:
        cv2.imwrite(base_dir + '/' + str(iteration) + '/{}_step02.jpg'.format(slide_id), img)
        cv2.imwrite(base_dir + '/' + str(iteration) + '/{}_step15.jpg'.format(slide_id), tissue_enclosed_marker_mask)
        cv2.imwrite(base_dir + '/intermediate/{}_step15.jpg'.format(slide_id), tissue_enclosed_marker_mask)

if __name__ == '__main__':

    base_dir = 'data/extractions'
    os.makedirs(os.path.join(base_dir, 'intermediate'), exist_ok=True)

    for iteration in range(4):
        # thumbnails_to_read = glob.glob('thumbnails/414950.svs_thumbnail.jpg')
        if iteration == 0:
            thumbnails_to_read = glob.glob('data/thumbnails/*.jpg')
        else:
            thumbnails_to_read = glob.glob('data/thumbnails/' + str(iteration) + '/*.jpg')

        os.makedirs(os.path.join('data/thumbnails', str(iteration + 1)), exist_ok=True)
        os.makedirs(os.path.join(base_dir, str(iteration)), exist_ok=True)
        
        # Extract markers
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
        tissue_masks = executor.map(extract_molecular_annotation, thumbnails_to_read)
        #for slide_id, tissue_mask in tqdm(tissue_masks):
        for thumbnail_id in tqdm(thumbnails_to_read):
            extract_molecular_annotation(thumbnail_id, base_dir, iteration)
        