import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import classification_report
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121,EfficientNetBN
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

train=pd.read_csv('../input/kidf-kpop-idol-dataset-female/kid_f_train.csv')
test=pd.read_csv('../input/kidf-kpop-idol-dataset-female/kid_f_test.csv')

class_names0=train['name'].unique().tolist()
tclass_names0=test['name'].unique().tolist()
available_class=list(set(class_names0)&set(tclass_names0))


traini = []
testi = []

for item in available_class :
    traini+=train[train['name']==item].index.tolist() # traini에 train의 index를 추가
    testi+=test[test['name']==item].index.tolist() # testi에 test의 index를 추가


train2 = train.iloc[traini] # train2에 traini의 index를 가진 train을 추가
test2 = test.iloc[testi] # test2에 testi의 index를 가진 test를 추가

train2 = train2.sample(frac=1, random_state=0) # train2를 랜덤으로 섞음
class_names = available_class # class_names에 available_class를 추가
num_class = len(class_names) # num_class에 class_names의 길이를 추가
N = list(range(num_class)) # N에 num_class의 range를 추가
normal_mapping = dict(zip(class_names, N)) # normal_mapping에 class_names와 N을 zip으로 묶음
reverse_mapping = dict(zip(N, class_names)) # reverse_mapping에 N과 class_names를 zip으로 묶음
train2['name'] = train2['name'].map(normal_mapping) # train2의 name을 normal_mapping으로 매핑
test2['name'] = test2['name'].map(normal_mapping) # test2의 name을 normal_mapping으로 매핑