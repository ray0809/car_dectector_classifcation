# car_dectector_classifcation
ssd is used for detecting and inception is used for classifying



# dataset
[cars dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)<br>
the dataset contain labels and rectangles<br>

# detector part
Here ssd is used for detecting
The code is written by [CarND-Vehicle-Detection](https://github.com/ksketo/CarND-Vehicle-Detection)


# classification part
Here InceptionV3 is used for classifying,the inputs come from ssd network<br>
training:8144 pics<br>
testing:8041<br>

# training
if you want to run the code,some path you should change to fit your own dataset<br>
First:run create_list.py<br>
Second:run train.py<br>

# test
run prediction.py



# examples
example1:
![pic1](https://github.com/ray0809/car_dectector_classifcation/blob/master/t.jpg)

example2:
![pic2](https://github.com/ray0809/car_dectector_classifcation/blob/master/t.jpg)


