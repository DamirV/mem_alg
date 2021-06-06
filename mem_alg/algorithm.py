import helper
import numpy as np
from scipy.ndimage.filters import generic_filter
from matplotlib import pyplot as plt
from scipy import stats
import cv2 as cv
# train n images => n * window_size ** 2 pieces
# test 10 * n * window_size ** 2 checks


class Calculator:
    def __init__(self, depth, window_size):
        self.memory = list()
        self.length = list()
        self.depth = depth
        self.window_size = window_size
        self.radius = self.window_size // 2

        for i in range(10):
            self.memory.append([])
            self.length.append([])

    def get_sobel(image):
        sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
        sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

        res = sobelx * sobelx + sobely * sobely
        res = np.sqrt(res)
        return res


    def find_coordinates(self, image):
        size = 28
        image = self.get_sobel(image)
        bright = 0
        coordinates = [0, 0]
        #helper.print_image(image[2 * self.radius: size - 2 * self.radius, 2 * self.radius:size - 2 * self.radius])

        for i in range(2*self.radius, size - 2*self.radius):
            for j in range(2*self.radius, size - 2*self.radius):

                tmp_sum = image[-self.radius + i: self.radius + i + 1, -self.radius + j: self.radius + j + 1].sum()
                if bright < tmp_sum:
                    bright = tmp_sum
                    coordinates[0] = j
                    coordinates[1] = i

        return coordinates


    def train(self, images, label):

        for i in range(len(images)):
            image = images[i]
            x, y = self.find_coordinates(image)
            #helper.print_image_with_dot(image, [y, x])

            for i in range(-self.radius, self.radius + 1):
                for j in range(-self.radius, self.radius + 1):
                    res = image[x + i - self.radius: x + i + self.radius + 1,
                          y + j - self.radius: y + j + self.radius + 1]
                    if res.sum() != 0:
                        self.memory[label].append(res)


    def check(self, image):
        y, x = self.find_coordinates(image)

        piece_of_image = image[x - self.radius: x + self.radius + 1, y - self.radius: y + self.radius + 1]
        tmp = []
        for i in range(10):
            tmp.append([])

        for i in range(10):
            if len(self.memory[i]) is not 0:
                for j in range(len(self.memory[i])):
                        tmp[i].append(np.linalg.norm(piece_of_image - self.memory[i][j]))

        res = np.zeros(10)
        for i in range(10):
            if len(self.memory[i]) is not 0:
                res[i] = int(min(tmp[i]))

        args = res.argsort()[:5]


        res = np.zeros(10)
        for i in args:
            res[i] = 1
        return res
