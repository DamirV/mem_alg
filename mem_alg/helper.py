from matplotlib import pyplot as plt
import numpy as np


def make_cloud():

    return


def distance(a, b):
    res = 0

    for i in range(len(a)):
        for j in range(len(a[0])):
            res += (int(a[i][j]) - int(b[i][j])) ** 2

    res = res ** 0.5
    return res


def cut(image, start_pos, window_size):
    res = []

    for i in range(window_size[0]):
        res.append([])
        for j in range(window_size[1]):
            res[i].append(image[start_pos[0] + i][start_pos[1] + j])

    return res


def print_image(image):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()
