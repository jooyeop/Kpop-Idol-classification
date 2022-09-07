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


train = []
testi = []