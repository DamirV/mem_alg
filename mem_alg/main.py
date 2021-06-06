from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import algorithm as al
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
window_size = 5

net = al.Calculator(1, window_size)

count = 100
for i in range(10):
    print(f"train {i}...")
    images = train[i][0:count]
    net.train(images, i)

for i in range(10):
    print(f"test {i}...")
    res = net.check(test[i][7])
    print(res)
    print("-----")









