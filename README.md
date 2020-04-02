# ComputerVision-CannyEdge

## Instructions
```shell script
pip3 install opencv-python
cd code
python main.py -s ../test_images/car/car.JPG -o ../output/car/car-out.png --sigma 1 --kernel_size 5
# or
python3 main.py -s ../test_images/car/car.JPG -o ../output/car/car-out.png --sigma 1 --kernel_size 5
# python2 won't work

# for multi patch double threshold
python main.py -s ../test_images/car/car.JPG -o ../output/car/car-out.png --sigma 1 --kernel_size 5 -mp -ps 5
```

```shell script
python main.py \
-s ../test_images/UofT/UofT.jpg \
-o ../output/UofT/UofT-out.png \
--sigma 1 \
--kernel_size 5 \
--low 0.01 \
--high 0.05 \
-mp -ps 5
```