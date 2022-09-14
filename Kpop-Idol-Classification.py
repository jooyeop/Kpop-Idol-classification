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

train_dir = '../input/kidf-kpop-idol-dataset-female/HQ_512x512/HQ_512x512'
test_dir = '../input/kidf-kpop-idol-dataset-female/test_final_with_degrad/test/degrad_image' 

image_files = []
image_labels=[]
for i in range(len(train2)):
    filename=train2.iloc[i,0]
    label=train2.iloc[i,1]
    image_files+=[os.path.join(train_dir, filename)]  # image_files에 train_dir과 filename을 합친 것을 추가
    image_labels+=[label] # image_labels에 label을 추가
    
timage_files = []
timage_labels = []
for i in range(len(test2)):
    tfilename=test2.iloc[i,0] # tfilename에 test2의 0번째 열을 추가
    tlabel=test2.iloc[i,1] # test2의 1번째 열을 tlabel에 추가
    timage_files+=[os.path.join(test_dir, tfilename)] # timage_files에 test_dir과 tfilename을 추가
    timage_labels+=[tlabel] # timage_labels에 tlabel을 추가
    
image_file_list = image_files # image_file_list에 image_files를 추가
image_label_list = image_labels # image_label_list에 image_labels를 추가
num_total = len(image_labels) # num_total에 image_labels의 길이를 추가

timage_file_list = timage_files # timage_file_list에 timage_files를 추가
timage_label_list = timage_labels # timage_label_list에 timage_labels를 추가
tnum_total = len(timage_labels) # tnum_total에 timage_labels의 길이를 추가

image_width, image_height = Image.open(image_file_list[0]).size # image_width와 image_height에 image_file_list의 0번째 열의 사이즈를 추가

print('Total image count:', num_total)
print("Image dimensions:", image_width, "x", image_height)
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])
print()

timage_width, timage_height = Image.open(timage_file_list[0]).size # timage_width와 timage_height에 timage_file_list의 0번째 열의 사이즈를 추가

print('Total image count:', tnum_total)
print("Image dimensions:", timage_width, "x", timage_height)
print("Label counts:", [len(timage_files[i]) for i in range(num_class)])


plt.subplots(3,3, figsize=(10,10))
for i,k in enumerate(np.random.randint(num_total, size=9)): # 9개의 랜덤한 숫자를 생성
    im = Image.open(image_file_list[k]) # im에 image_file_list의 k번째 열을 추가
    arr = np.array(im) # arr에 im을 array로 변환
    plt.subplot(3,3, i+1) # 3x3의 subplot을 생성
    plt.xlabel(class_names[image_label_list[k]]) # x축에 class_names의 image_label_list의 k번째 열을 추가
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255) # arr을 imshow로 출력
plt.tight_layout()
plt.show()


valid_frac = 0.2 # valid_frac에 0.2를 추가
trainX,trainY = [],[] # trainX와 trainY에 빈 리스트를 추가
valX,valY = [],[] # valX와 valY에 빈 리스트를 추가

for i in range(num_total):
    rann = np.random.random() # rann에 0~1 사이의 랜덤한 숫자를 추가
    if rann < valid_frac: # rann이 valid_frac보다 작으면
        valX.append(image_file_list[i]) # valX에 image_file_list의 i번째 열을 추가
        valY.append(image_label_list[i]) # valY에 image_label_list의 i번째 열을 추가
    else:
        trainX.append(image_file_list[i]) # trainX에 image_file_list의 i번째 열을 추가
        trainY.append(image_label_list[i]) # trainY에 image_label_list의 i번째 열을 추가

print(len(trainX),len(valX))



testX,testY = [],[]

for i in range(tnum_total):
    testX.append(timage_file_list[i]) # testX에 timage_file_list의 i번째 열을 추가
    testY.append(timage_label_list[i]) # testY에 timage_label_list의 i번째 열을 추가

print(len(testX))
print(testY[0:10])


trainX=np.array(trainX) # trainX를 array로 변환
trainY=np.array(trainY) # trainY를 array로 변환
valX=np.array(valX) # valX를 array로 변환
valY=np.array(valY) # valY를 array로 변환
testX=np.array(testX) # testX를 array로 변환
testY=np.array(testY) # testY를 array로 변환

# Define MONAI transforms, Dataset and Dataloader to pre-process data == 데이터 전처리를 위한 MONAI 변환, 데이터 세트 및 데이터로더 정의
class SumDimension(Transform):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, inputs):
        return inputs.sum(self.dim)

class MyResize(Transform):
    def __init__(self, size=(200,200)):
        self.size = size
    def __call__(self, inputs):
        image2=cv2.resize(np.array(inputs),dsize=(self.size[1],self.size[0]),interpolation=cv2.INTER_CUBIC) # image2에 cv2를 이용해 inputs를 resize
        return image2

train_transforms = Compose([
    LoadImage(image_only=True),
    Resize((-1,1)),
    SumDimension(2),
    MyResize(),
    AddChannel(),    
    ToTensor(),
]) # train_transforms에 Compose를 이용해 LoadImage, Resize, SumDimension, MyResize, AddChannel, ToTensor를 추가

val_transforms = Compose([
    LoadImage(image_only=True),
    Resize((-1,1)),
    SumDimension(2),
    MyResize(),
    AddChannel(),    
    ToTensor(),
]) # val_transforms에 Compose를 이용해 LoadImage, Resize, SumDimension, MyResize, AddChannel, ToTensor를 추가

act = Activations(softmax=True) # act에 softmax를 추가
to_onehot = AsDiscrete(to_onehot=num_class) # to_onehot에 AsDiscrete를 이용해 to_onehot을 num_class로 추가

class MedNISTDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files) # image_files의 길이를 반환

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index] # transforms를 이용해 image_files의 index번째 열을 반환하고 labels의 index번째 열을 반환


train_ds = MedNISTDataset(trainX, trainY, train_transforms)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

val_ds = MedNISTDataset(valX, valY, val_transforms)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

test_ds = MedNISTDataset(testX, testY, val_transforms)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=2)


device = torch.device("cuda:0")   #"cuda:0"

# EfficientNetBN
model = EfficientNetBN( 
    "efficientnet-b3",
    spatial_dims=2,            
    in_channels=1,
    num_classes=num_class
).to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
epoch_num = 40
val_interval = 1

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
auc_metric = ROCAUCMetric()
metric_values = list()

for epoch in range(epoch_num):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())         ##### .float()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(auc_result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), 'best_metric_model.pth')
                print('saved new best metric model')
                
            print(f"current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                  f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                  f" at epoch: {best_metric_epoch}")
            
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


plt.figure('train', (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Validation: Area under the ROC curve")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.show()

model.load_state_dict(torch.load('best_metric_model.pth'))
model.eval()
ty_true = list()
ty_pred = list()

with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            ty_true.append(test_labels[i].item())
            ty_pred.append(pred[i].item())


plt.subplots(3,3, figsize=(10,10))
for i,k in enumerate(np.random.randint(tnum_total, size=9)):
    im = Image.open(timage_file_list[k])
    arr = np.array(im) 
    plt.subplot(3,3, i+1)
    plt.xlabel(class_names[ty_pred[k]])
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report
print(classification_report(ty_true,ty_pred,target_names=class_names))