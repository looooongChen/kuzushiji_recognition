
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

line_plot = 1
fig,ax = plt.subplots(1)
with open('./detection.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    line_num = 0
    for row in readCSV:
        if line_num != line_plot:
            line_num += 1
            continue
        fname = row[0]
        print(fname)

        # im = imread(os.path.join('./dataset/train_images', fname+'.jpg'))
        im = np.array(Image.open(os.path.join('./dataset/test_images', fname+'.jpg')), dtype=np.uint8)
        ax.imshow(im)

        annotations = row[1].strip().split(' ')
        if len(annotations) < 5:
            break
        for i in range(len(annotations)):
            if i % 5 == 0:
                _, Xmin, Ymin, Xmax, Ymax = annotations[i], int(annotations[i+1]), int(annotations[i+2]), int(annotations[i+3]), int(annotations[i+4])
                rect = patches.Rectangle((Xmin, Ymin), Xmax-Xmin, Ymax-Ymin, linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)

        break
plt.axis('off')
plt.savefig("bbx_vis.png", bbox_inches='tight', dpi=600)
# plt.show()