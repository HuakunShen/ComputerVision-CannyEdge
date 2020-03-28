import cv2 as cv
import numpy as np


class CannyEdgeDetector:
    def __init__(self, sigma=1, kernel_size=1):
        self._images = {
            'source': None,
            'blurred': None,
            'edge': None,
            'theta': None
        }
        self.sigma = sigma
        self.kernel_size = kernel_size

    def read_image(self, filename, key):
        success = False
        msg = 'No Image Available'
        image = cv.imread(filename)
        if image is not None:
            self._images[key] = image
            success = True
            msg = "Read image " + filename + " Successfully\n"
        else:
            msg = "Failed to read image " + filename + "\n"
        return success, msg

    def write_image(self, filename, key):
        success = False
        msg = 'No Image Available'
        result = cv.imwrite(filename, self._images[key])
        if result:
            success = True
            msg = "Write image successfully to file " + filename + "\n"
        else:
            msg = "Failed to write image to file " + filename + "\n"

        return success, msg

    def gaussian_filter(self):
        x, y = np.mgrid[-self.kernel_size:self.kernel_size + 1, -self.kernel_size:self.kernel_size + 1].astype(
            np.float64)
        inverse_normalize_factor = (2 * np.pi * self.sigma ** 2)
        gaussian_kernel = np.exp(-((x ** 2 + y ** 2) / (2 * self.sigma ** 2)))
        return gaussian_kernel / inverse_normalize_factor

    def calculate_gradient(self):
        blur_image = self._images['blurred']
        edge_x = cv.Sobel(blur_image, cv.CV_64F, 1, 0, ksize=3)
        edge_y = cv.Sobel(blur_image, cv.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
        theta = np.arctan2(edge_y, edge_x)
        return edge, theta

    def non_maximum_suppression(self, angles):
        edge = self._images['edge']
        row, col = edge.shape[:2]
        result = np.zeros(edge.shape)
        angles[angles < 0] += 180
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                front = 0
                back = 0
                intensity = np.linalg.norm(edge[i, j])
                angle = angles[i, j]
                if 0 <= angle < 22.5 or 157.5 <= angle < 180:
                    front = np.linalg.norm(edge[i, j + 1])
                    back = np.linalg.norm(edge[i, j - 1])
                elif 22.5 <= angle < 67.5:
                    front = np.linalg.norm(edge[i - 1, j + 1])
                    back = np.linalg.norm(edge[i + 1, j - 1])
                elif 67.5 <= angle < 112.5:
                    front = np.linalg.norm(edge[i + 1, j])
                    back = np.linalg.norm(edge[i - 1, j])
                elif 112.5 <= angle < 157.5:
                    front = np.linalg.norm(edge[i - 1, j - 1])
                    back = np.linalg.norm(edge[i + 1, j + 1])
                if intensity >= front and intensity >= back:
                    result[i, j] = edge[i, j]
                else:
                    result[i, j] = 0
        self._images['suppress'] = result
        return result

    def double_threshold(self, low, high):
        if low >= 1 or high >= 1:
            print("low and high must be less than 0")
            return
        if low > high:
            print("low must be smaller than high")
            return
        image = self._images['suppress']
        max_intensity = image.max()
        high_threshold = max_intensity * high
        low_threshold = max_intensity * low
        row, col = image.shape
        result = np.zeros((row, col))
        strong_r, strong_c = np.where(image >= high_threshold)
        weak_r, weak_c = np.where((low_threshold < image) & (image < high_threshold))
        result[strong_r, strong_c] = 255
        result[weak_r, weak_c] = 20
        return result, 20, 255

    def hysteresis(self, weak, strong):
        surroundings = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        image = self._images['threshold']
        row, col = image.shape
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                if image[i, j] == weak:
                    strong_flag = 0
                    for surrounding in surroundings:
                        if image[i + surrounding[0]][j + surrounding[1]] == strong:
                            image[i, j] = strong
                            strong_flag = 1
                            break
                    if strong_flag == 0:
                        image[i, j] = 0
        return image

    def store_image(self, key, image):
        self._images[key] = image

    def get_image(self, key):
        return self._images[key]


def convolution(source_image, gaussian_kernel):
    row, col = source_image.shape[:2]
    out_image = np.zeros(source_image.shape)
    row_padding, col_padding = np.array(gaussian_kernel.shape) / 2

    for i in range(int(row_padding), int(row - row_padding)):
        for j in range(int(col_padding), int(col - col_padding)):
            conv_sum = 0
            for m in range(int(-row_padding), int(row_padding + 1)):
                for n in range(int(-col_padding), int(col_padding + 1)):
                    conv_sum += source_image[i + m, j + n] * gaussian_kernel[m, n]
            out_image[i, j] = conv_sum
    return out_image

