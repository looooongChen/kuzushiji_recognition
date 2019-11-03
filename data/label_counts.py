import csv
import numpy as np

top = 500

label_map = {}
counts = {}
with open('./train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    firstline = True
    for row in readCSV:
        if firstline:
            firstline = False
            continue

        annotations = row[1].split(' ')
        if len(annotations) < 5:
            continue
        for i in range(len(annotations)):
            if i % 5 == 0:
                label = annotations[i].strip()

                if label not in label_map:
                    label_map[label] = len(label_map) + 1
                    counts[label] = 1
                else:
                    counts[label] += 1

print("total classes: ", len(label_map)) 

labels = list(counts.keys())
label_num = list(counts.values())
label_num = np.array(label_num)
index = np.argsort(label_num)[::-1]

with open("label_counts.txt", "w") as text_file:
    for i in index:
        text_file.write(labels[i] + ' ' + str(label_num[i]) + '\n')


# import matplotlib.pyplot as plt
# cumulative = label_num[index]
# for i in range(1, len(cumulative)):
#     cumulative[i] = cumulative[i] + cumulative[i-1]
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(cumulative))+1, np.array(cumulative)/cumulative[-1])
# ax.set_xlabel('labels in descending order of frequency')
# ax.set_ylabel('cumulative frequency')

# x_major_ticks = np.arange(0, 4500, 500)
# x_minor_ticks = np.arange(0, 4500, 100)
# ax.set_xticks(x_major_ticks)
# ax.set_xticks(x_minor_ticks, minor=True)

# y_major_ticks = np.arange(0, 1, 0.1)
# y_minor_ticks = np.arange(0, 1, 0.02)
# ax.set_yticks(y_major_ticks)
# ax.set_yticks(y_minor_ticks, minor=True)

# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# ax.set_xlim([0,len(labels)])
# ax.set_ylim([0,1])

# plt.show()

