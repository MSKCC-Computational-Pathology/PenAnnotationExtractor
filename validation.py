# validation.py
# A tool to quantify the similarity between extracted pen annotated regions and manually annotated regions from WSI thumbnails.
# Created by PJ Schueffler, MSKCC
# schueffp@mskcc.org
#
# Schüffler PJ, Yarlagadda DVK, Vanderbilt C, Fuchs TJ: Overcoming an Annotation Hurdle: Digitizing Pen Annotations from Whole Slide Images. Journal of Pathology Informatics, 2021

import os
import imageio
import glob
import shutil
import numpy as np
import sklearn.metrics
from tqdm import tqdm
import math
import pandas as pd

dat = [['Run', 'ImageID', 'Dice', 'Jaccard', 'Precision', 'Recall', 'Kappa']]

dims = []

for i in range(4) :

    mask_pattern = "data/extractions/" + str(i) + "/*step15.jpg"

    for extraction_file in tqdm(glob.glob(mask_pattern)):

        # take the extraction mask
        imageID = os.path.basename(os.path.basename(extraction_file)).split('_')[1]

        # match with manual annotation
        annotation_file = "data/annotations/resized/labels_rs_" + imageID + ".png"

        # if both exist ...
        if (os.path.exists(annotation_file)) :

            # get the mask
            extraction_mask = imageio.imread(extraction_file)

            # get the annotation mask
            annotation_mask = 255-imageio.imread(annotation_file)[:,:,2]

            # verify dimensions
            if (annotation_mask.shape == extraction_mask.shape) :

                # save dimensions
                dims.append(extraction_mask.shape)

                mask1 = np.asarray(annotation_mask).astype(np.bool).flatten()
                mask2 = np.asarray(extraction_mask).astype(np.bool).flatten()

                # calculate Dice coefficiant = F-Score
                dice = sklearn.metrics.f1_score(mask1, mask2)

                # calculate Jaccard index = IoU
                jacc = sklearn.metrics.jaccard_score(mask1, mask2)

                # calculate Precision
                prec = sklearn.metrics.precision_score(mask1, mask2)

                # calculate Recall
                rec = sklearn.metrics.recall_score(mask1, mask2)

                # calcuate Kappa
                kappa = sklearn.metrics.cohen_kappa_score(mask1, mask2)
                
                dat.append([str(i), imageID, str(dice), str(jacc), str(prec), str(rec), str(kappa)])
            else:
                print("\nUnequal size: " + imageID)
        else:
            print("\nNo annotation file: " + imageID)

# show statistics of thumbnail dimensions
print(pd.DataFrame(dims).describe())

# show statistics of metrics
print(pd.DataFrame(dat[1:])[:][range(2,7)].astype(str).astype(float).describe())

# save the table
np.savetxt("metrics.csv", dat, delimiter=",", fmt='%s')
