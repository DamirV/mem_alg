from helper import print_image as pi
from helper import print_image_with_dot as piwd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import cv2 as cv
import math

def distance(x, y):
    c = math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    return c

class Saccades:
    def __init__(self, depth, window_size, step):
        self.dots = list()
        self.depth = depth
        self.window_size = window_size
        self.radius = self.window_size // 2
        self.step = step
        self.dot_count = -1
        for i in range(10):
            self.dots.append([])

    def get_sobel(self, image):
        sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
        sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

        res = sobelx * sobelx + sobely * sobely
        res = np.sqrt(res)
        return res


    def get_dot(self, image):
        size = 28
        image = self.get_sobel(image)
        bright = 0
        x = 0
        y = 0
        for i in range(size - 2*self.radius, 2*self.radius, -1):
            for j in range(2*self.radius, size - 2*self.radius):
                tmp_image = image[-self.radius + i: self.radius + i + 1, -self.radius + j: self.radius + j + 1]
                tmp_sum = tmp_image.sum()

                if bright < tmp_sum:
                    bright = tmp_sum
                    x = j
                    y = i

        return x, y


    def next_step(self, image, dot, saved_dots):
        area = 3
        image = self.get_sobel(image)
        bright = 0
        cur_x = dot[0]
        cur_y = dot[1]
        x = 0
        y = 0

        for i in range(-self.step, self.step+1):
            flag = False
            for saved_dot in saved_dots:
                if distance([cur_x + i, cur_y - self.step], saved_dot) <= area:
                    flag = True
            if flag:
                continue

            tmp_image = image[-self.radius + cur_x + i: self.radius + cur_x + 1 + i,
                        -self.radius + cur_y - self.step: self.radius + cur_y + 1 - self.step]

            tmp_sum = tmp_image.sum()

            if bright < tmp_sum:
                bright = tmp_sum
                x = cur_x + i
                y = cur_y - self.step

        for i in range(-self.step, self.step + 1):
            flag = False
            for saved_dot in saved_dots:
                if distance([cur_x + i, cur_y + self.step], saved_dot) <= area:
                    flag = True

            if flag:
                continue

            tmp_image = image[-self.radius + cur_x + i: self.radius + cur_x + 1 + i,
                        -self.radius + cur_y + self.step: self.radius + cur_y + 1 + self.step]
            tmp_sum = tmp_image.sum()

            if bright < tmp_sum:
                bright = tmp_sum
                x = cur_x + i
                y = cur_y + self.step

        for i in range(-self.step, self.step + 1):
            flag = False
            for saved_dot in saved_dots:
                if distance([cur_x - self.step, cur_y + i], saved_dot) <= area:
                    flag = True

            if flag:
                continue
            tmp_image = image[-self.radius + cur_x - self.step: self.radius + cur_x + 1 - self.step,
                        -self.radius + cur_y + i: self.radius + cur_y + 1 + i]
            tmp_sum = tmp_image.sum()

            if bright < tmp_sum:
                bright = tmp_sum
                x = cur_x - self.step
                y = cur_y + i

        for i in range(-self.step, self.step + 1):
            flag = False

            for saved_dot in saved_dots:
                if distance([cur_x + self.step, cur_y + i], saved_dot) <= area:
                    flag = True

            if flag:
                continue

            tmp_image = image[-self.radius + cur_x + self.step: self.radius + cur_x + 1 + self.step,
                        -self.radius + cur_y + i: self.radius + cur_y + 1 + i]
            tmp_sum = tmp_image.sum()

            if bright < tmp_sum:
                bright = tmp_sum
                x = cur_x + self.step
                y = cur_y + i

        return x, y


    def train(self, images, label):
        for i in range(len(images)):
            image = images[i]
            dot = self.get_dot(image)
            self.dots[label].append([])
            self.dot_count += 1

            self.dots[label][self.dot_count].append(dot)
            #piwd(image, dot)

            for j in range(self.depth):
                dot = self.next_step(image, dot, self.dots[label][self.dot_count])
                self.dots[label][self.dot_count].append(dot)
                #piwd(image, dot)

            print(f"{self.dot_count + 1} is done")

        return self.dots

    def check(self, image):
        dot = self.get_dot(image)
        tmp_dots = []
        tmp_dots.append(dot)
        for j in range(self.depth):
            dot = self.next_step(image, dot, tmp_dots)
            tmp_dots.append(dot)

        return tmp_dots
