
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pandas as pd

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('./unicode_translation.csv').values}

fontsize = 50
font = ImageFont.truetype('./NotoSerifCJKjp-Regular.otf', fontsize, encoding='utf-8')

def visualize_predictions(image_fn, labels):
    # Convert annotation string to array
    labels = np.array(labels.split(' ')).reshape(-1, 3)
    
    # Read image
    imsource = Image.open(image_fn).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y in labels:
        x, y = int(x), int(y)
        char = unicode_map[codepoint] # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x-10, y-10, x+10, y+10), fill=(255, 0, 0, 255))
        char_draw.text((x+25, y-fontsize*(3/4)), char, fill=(255, 0, 0, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.
    return np.asarray(imsource)

line_plot = 1
fig,ax = plt.subplots(1)
img_row = None
with open('./submission.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    line_num = 0
    for row in readCSV:
        if line_num != line_plot:
            line_num += 1
            continue
        img_row = row
        break
fname = img_row[0]
print(fname)

viz = visualize_predictions(os.path.join('./dataset/test_images', fname+'.jpg'), img_row[1])

plt.figure(figsize=(15, 15))
plt.imshow(viz, interpolation='lanczos')
plt.axis('off')
plt.savefig("res_vis.png", bbox_inches='tight', dpi=600)
# plt.show()
