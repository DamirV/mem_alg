import helper
import numpy as np
from scipy.ndimage.filters import generic_filter
from matplotlib import pyplot as plt
from scipy import stats

# train n images => n * window_size ** 2 pieces
# test 10 * n * window_size ** 2 checks

class Calculator4:

    def __init__(self, depth, start_pos, window_size):
        self.memory = list()
        self.length = list()
        self.mean = 0
        self.depth = depth
        self.window_size = window_size
        self.radius = self.window_size // 2
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.label = ""
        self.image = 0

        for i in range(10):
            self.memory.append([])
            self.length.append([])


    def get_sobel(self, image):
        def sobel_filter(P):
            return (np.abs((-P[0] + P[3]) + np.abs(-P[1] + P[2])))

        image = generic_filter(image, sobel_filter, (2, 2))

        #helper.print_image(image)###

        return image


    def find_coordinates(self):
        size = 28
        image = self.get_sobel(self.image)

        bright = 0
        coordinates = [0, 0]
        for i in range(self.radius, size - self.radius):
            for j in range(self.radius, size - self.radius):
                sum = 0
                for y in range(-self.radius, self.radius + 1):
                    for x in range(-self.radius, self.radius + 1):
                        sum += image[j + x, i + y]

                if bright < sum:
                    bright = sum
                    coordinates[0] = i
                    coordinates[1] = j

        return coordinates

    def train(self, images, label):

        #helper.print_image(image)

        for i in range(len(images)):
            self.image = images[i]
            coordinates = self.find_coordinates()

            """
            plt.imshow(self.image, cmap=plt.get_cmap('gray'))###
            plt.scatter(coordinates[0], coordinates[1], c='red', marker='o')###
            plt.show()###
            """

            self.x = coordinates[1]
            self.y = coordinates[0]

            """
            print(coordinates)
            helper.print_image(image[self.x - self.radius: self.x + self.radius + 1,
                                self.y - self.radius: self.y + self.radius + 1])###
            """

            for i in range(-self.radius, self.radius + 1):
                for j in range(-self.radius, self.radius + 1):
                    if self.x + i - self.radius < 0 or self.x + i + self.radius + 1 > 28 \
                            or self.y + j - self.radius < 0 or self.y + j + self.radius + 1 > 28:
                        break

                    res = self.image[self.x + i - self.radius: self.x + i + self.radius + 1,
                          self.y + j - self.radius: self.y + j + self.radius + 1]

                    # helper.print_image(res) ###
                    self.memory[label].append(res)


    def check(self, image):
        self.image = image
        coordinates = self.find_coordinates()
        self.x = coordinates[1]
        self.y = coordinates[0]

        #plt.imshow(self.image, cmap=plt.get_cmap('gray'))  ###
        #plt.scatter(coordinates[0], coordinates[1], c='red', marker='o')  ###
        #plt.show()  ###

        piece_of_image = image[self.x - self.radius: self.x + self.radius + 1,
                         self.y - self.radius: self.y + self.radius + 1]

        #helper.print_image(piece_of_image)

        tmp = []
        for i in range(10):
            tmp.append([])

        for i in range(10):
            if len(self.memory[i]) is not 0:
                for j in range(len(self.memory[i])):
                        tmp[i].append(np.linalg.norm(piece_of_image - self.memory[i][j]))

        """
        for i in range(10):
            print(tmp[i])
            print(np.array(tmp[i]).mean())
            print(np.array(tmp[i]).min())
            print("-----------")
        """

        res = np.zeros(10)
        for i in range(10):
            if len(self.memory[i]) is not 0:
                res[i] = int(min(tmp[i]))

        args = res.argsort()[:3]
        res = np.zeros(10)
        for i in args:
            res[i] = 1
        return res
