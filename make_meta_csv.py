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
from pandas.io.pytables import IndexCol
! pip install gputil
from utils.helper import *
import cv2, glob


def imread(path):
    image = cv2.imread(path)
    return image


ilist = glob.glob('../data/**/*.png',recursive=True)

def parse_(path):
    _,_,data_type,ttg,bad_type,filename= path.split('/')
    return data_type,ttg,bad_type,filename


ilist[10].split('/')
parse_(ilist[10])




'''
pandas row =-> data frame 
df = pd.DataFrame([[1, 2], [3, 4]], columns = ["a", "b"])
'''

rows = []
for i, path in enumerate(ilist):
    id = f'ID_{i:06d}'
    data_type,ttg,bad_type,filename = parse_(path)
    rows.append([id,data_type,ttg,bad_type,filename,path])

df = pd.DataFrame( rows, columns=['id','data_type','ttg_type','error_type','filename','path'])

df.to_csv('meta.csv')


df = pd.read_csv('meta.csv',index_col=0)



'''
pandas row sliceing 예제 
df.loc[df.index.isin(['one','two'])]
'''


print(df.columns)
pd.unique(df['data_type'])
pd.unique(df['ttg_type'])
pd.unique(df['error_type'])

cond = df['data_type'].isin(['pill']) & \
       df['ttg_type'].isin(['train'])

sub = df.loc[cond]
print(sub)
