from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import algorithm3 as al
import helper as hp
import loader
from scipy import stats


def testF(net, image, label):
    tmp = net.check(image)

    print("---------------------------------")
    print(f"label = {label}")
    print(f"mean = {np.array(tmp).mean()}")
    print(f"sigma = {np.array(tmp).var() ** 0.5}")
    print(f"max = {max(tmp)}")
    print(f"min = {min(tmp)}")
    print(f"percentile 50% = {np.percentile(tmp, 50)}")


train, test = loader.load()
size = 60000
start_pos = (10, 10)
window_size = 5

net = al.Calculator3(1, start_pos, window_size)

count = 1000
for i in range(10):
    print(f"train {i}...")
    images = train[i][0:count]
    net.train(images, i)

for i in range(10):
    print(f"test {i}...")
    res = net.check(test[i][7])
    print(res)
    print("-----")


"""
image = train[0][7]

plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.scatter(coordinates[0], coordinates[1], c='red', marker='o')
plt.show()

plt.imshow(net.get_sobel(image), cmap=plt.get_cmap("gray"))
plt.scatter(coordinates[0], coordinates[1], c='red', marker='o')
plt.show()
"""








