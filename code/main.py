import argparse

parser = argparse.ArgumentParser(description='Canny Edge Detection Arugument parser.')
parser.add_argument('-s', '--source', help='Source Image Address', required=True, type=str)
parser.add_argument('-o', '--output', help='Output Image Address', required=True, type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.source)
    print(args.output)