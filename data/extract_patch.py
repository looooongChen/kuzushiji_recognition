import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import os
import uuid 

# top_N = 2000

# label_counts = {}

# with open("label_counts.txt", "r") as lines:
#     for ind, line in enumerate(lines, start=1):
#         if ind > top_N:
#             break
#         parsed = line.strip().split(' ')
#         label_counts[parsed[0]] = int(parsed[1])

# print(label_counts, len(label_counts))

save_dir = './chars'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
labels = []
PATCH_SZ = (64, 64)

patch_dict = {}
line_num = 0
with open('./train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if line_num == 0:
            line_num += 1
            continue
        # if line_num == 10:
        #     break
        fname = row[0]
        print(line_num, fname)
        line_num += 1

        im = imread(os.path.join('./train_images', fname+'.jpg'))

        annotations = row[1].strip().split(' ')
        if len(annotations) < 5:
            continue
        for i in range(len(annotations)):
            if i % 5 == 0:
                l, x, y, w, h = annotations[i], int(annotations[i+1]), int(annotations[i+2]), int(annotations[i+3]), int(annotations[i+4])
                l = l.strip()
                # if not os.path.exists(os.path.join(save_dir, l)):
                #     os.mkdir(os.path.join(save_dir, l))
                patch = resize(im[y:y+h, x:x+w],PATCH_SZ, preserve_range=True).astype(np.uint8)
                if l in patch_dict.keys():
                    patch_dict[l].append(patch)
                else:
                    patch_dict[l] = [patch]
                
# for k, patch in patch_dict.items():
#     np.save(os.path.join(save_dir, k+'.npy'), np.array(patch)) 
                
