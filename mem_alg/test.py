from keras.datasets import mnist
import numpy as np
import helper
import loader
from scipy.ndimage.filters import generic_filter
import saccades
import math
from scipy import ndimage, misc
import cv2 as cv

train, test = loader.load()

net = saccades.Saccades(5, 5, 5)
number = 0
size = 1000
imgs = train[0][number: number + size]
new_dot = net.train(imgs, 0)
