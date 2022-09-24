# Kpop-Idol-Classification
Monai 이미징 프레임 워크를 이용한 K-pop 아이돌 사진구분

## 프로젝트 목적
K-pop 아이돌의 사진을 구분하는 프로젝트를 진행

## 프로젝트 배경
이미징 프레임워크 이해력 상승

## 연구 및 개발에 필요한 데이터 셋 소개
https://www.kaggle.com/datasets/vkehfdl1/kidf-kpop-idol-dataset-female

캐글에 등록되어있는 K-pop 아이돌 사진을 활용한 프로젝트 진행

## 연구 및 개발에 필요한 기술 스택
Monai => 엔비디아 MONAI 이미징 프레임워크
    
## 해당 기술(또는 연구방법)의 최근 장점과 단점
- Monai를 사용한 장점과 단점
    장점
      → 엔비디아에서 출시한 의료용 모델을 활용한 이미징 프레임워크
      해당 모델을 이용하여 프로젝트를 진행했을 때 장점과 단점에 대해선 조금 더 찾아서 보완할 예정
      
```Python3
# EfficientNetBN
model = EfficientNetBN("efficientnet-b3",spatial_dims=2, in_channels=1,num_classes=num_class).to(device)

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
 ```


## 결과
##모델의 이미지 분류 전 아이돌 클래스 구분
![Figure_1](https://user-images.githubusercontent.com/97720878/192100943-bf03cd24-a627-46b6-b814-e86308ea2202.png)

##모델 구축 후 평균 Loss 및 ROC Curve
(좌) Loss (우) ROC Curve
![ROC,Loss](https://user-images.githubusercontent.com/97720878/192100969-420aa843-4ac8-4699-9d52-e6caa61438df.png)

##모델 구축 후 이미지 클래스 분류
![Figure_2](https://user-images.githubusercontent.com/97720878/192100994-80e63340-3edc-429b-a5f4-9f472cad1943.png)


## 한계점 및 해결 방안
참고코드를 활용한 프로젝트가 아닌 직접 모델을 구축해보는것이 목표
참고 코드 : https://www.kaggle.com/code/stpeteishii/kpop-idol-classify-monai-pytorch

또한 데이터셋도 직접 구축해보는것이 목표
