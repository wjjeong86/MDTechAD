'''
data 폴더 구조
data
+-bottle
  +-ground_truth
  +-train
    +-good
  +-test
    +-good
    +-broken_large
    +- ...
+-pill
+-...


meta 파일 구조


id 데이터분야(bottle/pill) good/bad good/badtype 파일위치



''' 




import pandas as pd, numpy as np
! pip install gputil
from utils.helper import *
import cv2, glob


def imread(path):
    image = cv2.imread(path)
    return image


ilist = glob.glob('../data/**/*.png',recursive=True)

glob.