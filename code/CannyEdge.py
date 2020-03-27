import numpy as np
import cv2 as cv


class CannyEdge:
    def __init__(self):
        self._images = {
            'source': None,
            'output': None
        }

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
