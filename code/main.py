import argparse
import sys
from edgeDetector import *
import os


def parse_arguments(argv, prog=''):
    parser = argparse.ArgumentParser(prog, description='Canny Edge Detection Arugument parser.')
    parser.add_argument('-s', '--source', help='Source Image Address', required=True, type=str)
    parser.add_argument('-o', '--output', help='Output Image Address', required=True, type=str)
    parser.add_argument('-q', '--quiet', action='store_true', help='print quite')
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose')
    parser.add_argument('--sigma', help='sigma value', default=1, type=int)
    parser.add_argument('--kernel_size', help='kernel size value', default=1, type=int)
    args, unprocessed_argv = parser.parse_known_args(argv)
    print("Source Image: {}".format(args.source))
    print("Output Image: {}".format(args.output))
    print("unprocessed_argv: {}".format(unprocessed_argv))
    return args, unprocessed_argv


def main(argv, prog=''):
    args, unprocessed_argv = parse_arguments(argv, prog)
    # parse output file name
    parsed_output = os.path.splitext(args.output)
    ed = CannyEdgeDetector(sigma=args.sigma, kernel_size=args.kernel_size)
    g_filter = ed.gaussian_filter()
    success, msg = ed.read_image(args.source, 'source')
    if not success:
        print(msg)
        exit(1)
    image = np.array(ed.get_image('source'))
    if image.shape[2] > 1:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    out = convolution(image, g_filter)
    ed.store_image('blurred', out)
    success, msg = ed.write_image(parsed_output[0] + '-blur' + parsed_output[1], 'blurred')
    if not success:
        print(msg)
        exit(1)
    edge, theta = ed.calculate_gradient()
    ed.store_image('edge', edge)
    success, msg = ed.write_image(parsed_output[0] + '-edge' + parsed_output[1], 'edge')
    if not success:
        print(msg)
        exit(1)
    ed.non_maximum_suppression(theta * 180 / np.pi)
    success, msg = ed.write_image(parsed_output[0] + '-suppress' + parsed_output[1], 'suppress')
    if not success:
        print(msg)
        exit(1)
    threshold, weak, strong = ed.double_threshold(0.03, 0.1)
    ed.store_image('threshold', threshold)
    success, msg = ed.write_image(parsed_output[0] + '-threshold' + parsed_output[1], 'threshold')
    if not success:
        print(msg)
        exit(1)
    result = ed.hysteresis(weak, strong)
    ed.store_image('result', result)
    success, msg = ed.write_image(args.output, 'result')
    if not success:
        print(msg)
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:], sys.argv[0])
