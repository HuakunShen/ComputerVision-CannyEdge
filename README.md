# ComputerVision-CannyEdge

## Instructions

The code is written in python3, the opencv library is also a python3 version. Running it with python2 and python2's opencv is not guaranteed to work.

```shell script
pip3 install opencv-python
cd code
python main.py -s ../test_images/car/car.JPG -o ../output/car/car-out.png --sigma 1 --kernel_size 1 --low 0.1 --high 0.3
# python2 won't work

# for multi patch double threshold
python main.py -s ../test_images/car/car.JPG -o ../output/car/car-out.png --sigma 1 --kernel_size 1 --low 0.1 --high 0.3 -mp -ps 100
# if you want a image with patches indicated in image to be saved, add --show_patch flag

# for pyramid blurring
python main.py -s ../test_images/car/car.JPG -o ../output/car/car-out.png --sigma 1 --kernel_size 1 --low 0.1 --high 0.3 --pyramid
```

The input/output folder/files have to exist in order for program to read and write. A folder doesn't exist wouldn't be created in the output process, and the program will stop. 