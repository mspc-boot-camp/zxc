import requests
import os
import numpy as np
import re
import cv2
from threading import Thread,Lock
import matplotlib.pyplot as plt

path=r'F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/input/'
def read_picture():
    with open(path+r'image.txt') as f:
        path_lines = f.readlines()
    return path_lines

'''
files=read_picture()
print(files)
print(files[1])
print(str(list(files[1])[-8:-10]))
s='dog/2.jpg'
folder = s.split('/')[-1].split('.')[0]
image = s.split('/')[-1]

print(os.path.join(path, 'validation', folder, image))
print(folder)
print(image)
'''