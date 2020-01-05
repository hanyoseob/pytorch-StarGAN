import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# dir_result = './result/stargan/celeba/images'
dir_result = './result/stargan/rafd/images'
lst_result = os.listdir(dir_result)

# np.random.shuffle(lst_result)

nx = 128
ny = 128
nch = 3

n = 8
# m = 1 + 5
m = 1 + 8

m_id = 50

img = torch.zeros((n*m, ny, nx, nch))

for i in range(m):
    for j in range(n):
        p = i + m*(j + m_id)
        q = n * i + j

        img[q, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_result[p]))[:, :, :nch])

img = img.permute((0, 3, 1, 2))

plt.figure(figsize=(n, m))
plt.axis("off")
# plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))

plt.show()

