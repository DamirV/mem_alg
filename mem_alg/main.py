from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import algorithm as al
import helper as hp
import loader
from scipy import stats

train, test = loader.load()

size = 60000
start_pos = (10, 10)
window_size = (10, 10)


dist = []
chunks = []
n = 100
for i in range(n):
    chunks.append(hp.cut(train[0][i], window_size=window_size, start_pos=start_pos))

for i in range(n):
    for j in range(n):
        dist.append(hp.distance(chunks[i], chunks[j]))

print(f"mean: {np.array(dist).mean()}")
print(f"sigma: {np.array(dist).var() ** 0.5}")

plt.hist(dist)
plt.show()

chunk1 = hp.cut(train[0][0], window_size=window_size, start_pos=start_pos)
chunk2 = hp.cut(train[0][12], window_size=window_size, start_pos=start_pos)
dist = hp.distance(chunk1, chunk2)

hp.print_image(chunk1)
hp.print_image(chunk2)

print(dist)




"""
plt.figure(1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(digits[1][i], cmap=plt.get_cmap('gray'))


plt.figure(2)
a = digits[1][0]
plt.imshow(a, cmap=plt.get_cmap('gray'))


plt.figure(3)
image = hp.cut(a, start_pos, window_size)
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()
"""





