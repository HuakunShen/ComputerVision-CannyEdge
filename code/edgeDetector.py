import cv2 as cv
import numpy as np


class CannyEdgeDetector:
    def __init__(self, sigma=1, kernel_size=1, show_patch_border=False):
        self._images = {
            'source': None,
            'blurred': None,
            'edge': None,
            'theta': None,
            'patches': None
        }
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.show_patch_border = show_patch_border

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
        gaussian_kernel = gaussian_kernel / inverse_normalize_factor
        gaussian_kernel /= gaussian_kernel.sum()
        return gaussian_kernel

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
                intensity = edge[i, j]
                angle = angles[i, j]
                if 0 <= angle < 22.5 or 157.5 <= angle < 180:
                    front = edge[i, j + 1]  # right pixel
                    back = edge[i, j - 1]  # left pixel
                elif 22.5 <= angle < 67.5:
                    front = edge[i - 1, j + 1]  # top right pixel
                    back = edge[i + 1, j - 1]  # bottom left pixel
                elif 67.5 <= angle < 112.5:
                    front = edge[i + 1, j]  # bottom pixel
                    back = edge[i - 1, j]  # top pixel
                elif 112.5 <= angle < 157.5:
                    front = edge[i - 1, j - 1]  # top left pixel
                    back = edge[i + 1, j + 1]  # bottom right pixel
                if intensity >= front and intensity >= back:
                    result[i, j] = edge[i, j]  # if this pixel is the brightest among neighbors
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

    def multi_patch_double_threshold(self, low, high, patch_size):
        if patch_size > min(self._images['suppress'].shape[:2]):
            return self.double_threshold(low, high)  # patch size too large, ignore multi-patch
        if low >= 1 or high >= 1:
            print("low and high must be less than 0")
            exit(1)
        if low > high:
            print("low must be smaller than high")
            exit(1)
        image = self._images['suppress']
        n_row, n_col = image.shape
        result = np.zeros((n_row, n_col))
        patch_border_pixels = [[], []]
        # divide into patches
        row_s, row_e, col_s, col_e = 0, 0, 0, 0  # s for start, e for end
        while row_e < n_row:
            row_e = row_s + patch_size if row_s + patch_size < n_row else n_row
            col_s, col_e = 0, 0
            while col_e < n_col:
                col_e = col_s + patch_size if col_s + patch_size < n_col else n_col
                patch = image[row_s:row_e, col_s:col_e].copy()  # copy a patch from the image
                if self.show_patch_border:
                    rows, cols = patch_border(row_s, row_e, col_s, col_e)
                    patch_border_pixels[0].extend(rows)
                    patch_border_pixels[1].extend(cols)
                max_intensity = patch.max()  # obtain local max intensity
                high_threshold = max_intensity * high
                low_threshold = max_intensity * low
                strong_r, strong_c = np.where(patch >= high_threshold)  # find strong pixels with high threshold
                weak_r, weak_c = np.where(
                    (low_threshold < patch) & (patch < high_threshold))  # find weak pixels with thresholds
                # update patch pixels intensity
                tmp = np.zeros(patch.shape)
                tmp[strong_r, strong_c] = 255
                tmp[weak_r, weak_c] = 20
                # copy patch pixels to result image
                result[row_s:row_e, col_s:col_e] = tmp
                col_s = col_e
            row_s = row_e
        return result, 20, 255, patch_border_pixels

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
    row_padding, col_padding = np.array(gaussian_kernel.shape) // 2

    for i in range(int(row_padding), int(row - row_padding)):
        for j in range(int(col_padding), int(col - col_padding)):
            patch = source_image[i - row_padding:i + row_padding + 1, j - col_padding:j + col_padding + 1]
            out_image[i, j] = int((np.flip(patch) * gaussian_kernel).sum())
    return out_image


def patch_border(row_s, row_e, col_s, col_e):
    n_row, n_col = row_e - row_s, col_e - col_s
    rows = []
    cols = []
    # left border
    rows.extend(np.arange(row_s, row_e))
    cols.extend(np.repeat(col_s, n_row))
    # right border
    rows.extend(np.arange(row_s, row_e))
    cols.extend(np.repeat(col_e - 1, n_row))
    # top border
    rows.extend(np.repeat(row_s, n_col))
    cols.extend(np.arange(col_s, col_e))
    # bottom border
    rows.extend(np.repeat(row_e - 1, n_col))
    cols.extend(np.arange(col_s, col_e))
    return rows, cols
