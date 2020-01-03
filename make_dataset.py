import os
import numpy as np
import matplotlib.pyplot as plt

# fig, (splt1, splt2) = plt.subplots(1, 2)

sz = 256

dir_data = '../datasets/rafd/'
lines = os.listdir(dir_data)

all_attr_names = ['angry', 'contemptuous', 'disgusted', 'fearful',
                  'happy', 'neutral', 'sad', 'surprised']

attr2idx = {}
idx2attr = {}

for i, attr_name in enumerate(all_attr_names):
    attr2idx[attr_name] = i
    idx2attr[i] = attr_name

# self.attr2idx = attr2idx
# self.idx2attr = idx2attr

np.random.seed(1234)
np.random.shuffle(lines)

train_dataset = []

for i, line in enumerate(lines):
    label = np.zeros(len(all_attr_names), dtype=np.float32)
    split = line.split('_')
    filename = line
    attr = split[4]

    label[attr2idx[attr]] = 1

    train_dataset.append([filename, label])

print('make db')

# for i_, lst_ in enumerate(lst_data):
#     data_ = plt.imread(os.path.join(dir_data, lst_))
#
#     label_ = data_[:, :sz, :]
#     input_ = data_[:, sz:, :]
#
#     np.save(os.path.join(dir_data, "label_%05d.npy" % i_), np.float32(label_))
#     np.save(os.path.join(dir_data, "input_%05d.npy" % i_), np.float32(input_))
