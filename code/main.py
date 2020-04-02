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
    parser.add_argument('-mp', '--multiple_patch', action='store_true',
                        help='Indicate whether multiple-patch modification should be applied')
    parser.add_argument('-ps', '--patch_size', type=int, default=None,
                        help='patch size, total number of pixels will be patch_size^2')
    parser.add_argument('--low', type=float, default=0.01, help='low threshold')
    parser.add_argument('--high', type=float, default=0.1, help='high threshold')
    parser.add_argument('-p', '--pyramid', action='store_true', help='use pyramid for blurring')
    args, unprocessed_argv = parser.parse_known_args(argv)
    print("Source Image: {}".format(args.source))
    print("Output Image: {}".format(args.output))
    print("unprocessed_argv: {}".format(unprocessed_argv))
    return args, unprocessed_argv


def args_check(args):
    # args check for multiple patch:
    if args.multiple_patch:
        if not args.patch_size:
            print("Multiple patch selected but patch size not selected")
            exit(1)


def main(argv, prog=''):
    args, unprocessed_argv = parse_arguments(argv, prog)
    args_check(args)
    # parse output file name
    parsed_output = os.path.splitext(args.output)
    filename_setting_addtion = ''
    if args.multiple_patch:
        filename_setting_addtion += '-multiple_patch'
    ed = CannyEdgeDetector(sigma=args.sigma, kernel_size=args.kernel_size)
    g_filter = ed.gaussian_filter()
    print('Reading source image')
    success, msg = ed.read_image(args.source, 'source')
    if not success:
        print(msg)
        exit(1)
    image = np.array(ed.get_image('source'))
    if image.shape[2] > 1:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if args.pyramid:
        print('Blur image, apply pyramid to reduce noise')
        out = cv.pyrDown(image)
    else:
        print('Blur image, apply gaussian filter to reduce noise')
        out = convolution(image, g_filter)
    ed.store_image('blurred', out)
    success, msg = ed.write_image(parsed_output[0] + '-blur' + filename_setting_addtion + parsed_output[1], 'blurred')
    if not success:
        print(msg)
        exit(1)

    print('Calculating gradient and theta')
    edge, theta = ed.calculate_gradient()
    ed.store_image('edge', edge)
    success, msg = ed.write_image(parsed_output[0] + '-edge' + filename_setting_addtion + parsed_output[1], 'edge')
    if not success:
        print(msg)
        exit(1)

    print('Non-Max Suppression')
    ed.non_maximum_suppression(theta * 180 / np.pi)
    success, msg = ed.write_image(parsed_output[0] + '-suppress' + filename_setting_addtion + parsed_output[1],
                                  'suppress')
    if not success:
        print(msg)
        exit(1)
    # start multiple patch modification
    print('Double Threshold')
    if args.multiple_patch:
        threshold, weak, strong = ed.multi_patch_double_threshold(args.low, args.high, args.patch_size)
    else:
        threshold, weak, strong = ed.double_threshold(args.low, args.high)
    ed.store_image('threshold', threshold)
    success, msg = ed.write_image(parsed_output[0] + '-threshold' + filename_setting_addtion + parsed_output[1],
                                  'threshold')
    if not success:
        print(msg)
        exit(1)

    print('Hysteresis: Keep only the pixels that belong to, or are connected to "strong edges')
    result = ed.hysteresis(weak, strong)
    ed.store_image('result', result)
    success, msg = ed.write_image(args.output, 'result')
    if not success:
        print(msg)
        exit(1)
    print("Done")


if __name__ == "__main__":
    main(sys.argv[1:], sys.argv[0])

    # for debug only
    # main(['-s', '../test_images/car/car.JPG', '-o', '../output/car/car-out.png', '--sigma', '1', '--kernel_size', '5',
    #       '-mp', '-ps', '5'])
    # python main.py -s ../test_images/car/car.JPG -o ../output/car/car-out.png --sigma 1 --kernel_size 5
