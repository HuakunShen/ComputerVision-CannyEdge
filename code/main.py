import argparse
import sys
from CannyEdge import *


def parse_arguments(argv, prog=''):
    parser = argparse.ArgumentParser(prog, description='Canny Edge Detection Arugument parser.')
    parser.add_argument('-s', '--source', help='Source Image Address', required=True, type=str)
    parser.add_argument('-o', '--output', help='Output Image Address', required=True, type=str)
    parser.add_argument('-q', '--quiet', action='store_true', help='print quite')
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose')
    args, unprocessed_argv = parser.parse_known_args(argv)
    print("Source Image: {}".format(args.source))
    print("Output Image: {}".format(args.output))
    print("unprocessed_argv: {}".format(unprocessed_argv))
    return args, unprocessed_argv


def main(argv, prog=''):
    canny_edge = CannyEdge()
    args, unprocessed_argv = parse_arguments(argv, prog)
    # read input image
    ok, msg = canny_edge.read_image(args.source, 'source')
    if not ok:
        print('Error: read_image: ', msg)
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:], sys.argv[0])
