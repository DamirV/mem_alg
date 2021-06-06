from matplotlib import pyplot as plt
import numpy as np


def distance(a, b):
    res = np.linalg.norm(np.array(a) - np.array(b))
    return res


def cut(image, start_pos, window_size):
    res = []

    for i in range(window_size):
        res.append([])
        for j in range(window_size):
            res[i].append(image[start_pos[0] + i][start_pos[1] + j])

    return res


def print_image(image):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


def print_image_with_dot(image, crd):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.scatter(crd[0], crd[1], c='red', marker='o')
    plt.show()


def print_images(image1, image2):

    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)

    fig.add_subplot(1, 2, 2)
    plt.imshow(image2)

    plt.show()
