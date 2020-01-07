
# 如何使用

## 直接进行预测

1. 清空并将将预测的数据放入```prediction```文件夹
2. 运行```python predict.py```
3. 结果在```prediction.csv```中

## 重新训练

1. 将要训练的数据集依照文件说明放入```img```下的```cat```, ```no_cat```文件夹。要求对不同种类的猫进行分类并放于```cat```下的不同的文件夹之下
2. 在当前目录下运行 ```python Cat_Detection_and_Classification.py```
    +__注意: 训练强制使用需要GPU__  
3. 获取```*.h5```文件，其中
    + ```Detection.h5```为二分类的权重以及模型文件
    + ```Detection_weights.h5```为二分类的权重文件
    + ```Identification.h5```为种类识别(多分类)的权重以及模型文件
    + ```Identification_weights.h5```为种类识别(多分类)的权重文件
    + 
# 文件目录说明

cat-recognition</br>
├─ 3355_130_weights.h5</br>
├─ [README.md](./readme.md) //README文件</br>
├─ SIFT.ipynb</br>
├─ binary</br>
│	├─ .ipynb_checkpoints</br>
│	├─ Classification (cats & dogs).ipynb	//tensorflow 教程</br>
│	├─ [Dog Detection (BN).ipynb](./binary/Dog&#32;Detection&#32;(BN).ipynb)	//狗的识别，批正则化</br>
│	├─ [Dog Detection (Dropout) .ipynb](./binary/Dog&#32;Detection&#32;(Dropout)&#32;.ipynb)	//狗的识别，尝试不同的Dropout</br>
│	├─ [Dog Detection (other kernel size).ipynb](./binary/Dog&#32;Detection&#32;(other&#32;kernel&#32;size))	//狗的识别，不同的卷积核</br>
│	├─ Try_cat_dog.ipynb	//一般猫狗分类</br>
│	├─ binary_model.png	//结果统计图</br>
│	├─ bn.png	//结果统计图</br>
│	├─ droup_out_0.8_0.8.png	//结果统计图</br>
│	├─ lr0.0005.png	//结果统计图</br>
│	├─ lr0.002.png	//结果统计图</br>
│	└─ no_dropout.png	//结果统计图</br>
├─ dataCleansing.ipynb</br>
├─ [img](./img/)	//数据集</br>
│	├─ cat	//15种猫分类</br>
│	└─ no_cat	//无猫</br>
├─ img_sub	//15种猫的数据集</br>
│	└─ cat	//~每种200张图片</br>
│	└─ no_cat	//~每种200张图片</br>
├─ multiclass</br>
│	├─ 150_3355kernal.png	//结果统计图</br>
│	├─ Classification (clothes).ipynb	//tensorflow多分类教程</br>
│	├─ cnn-lgb.ipynb	//cnn+xgboost/lsm 结果很差，实验性质，但是期望应该是比CNN直接要好，仅作参考</br>
│	├─ cat_dataset.png	//结果统计图</br>
│	├─ [cat_others_multi.ipynb](./multiclass/cat_others_multi.ipynb)	//多分类</br>
│	├─ cd-0.25dp.png	//结果统计图</br>
│	├─ f_3355_90.png	//结果统计图</br>
│	├─ sub_cd_0.25dp.png	//结果统计图</br>
│	└─ sub_cd_0.25dp_150.png	//结果统计图</br>
├─ [report.md](./report.md) //总结报告</br>
├─ report_en.md</br>
├─ [web-scraping.ipynb](./web-scraping.ipynb) //拉取图片的爬虫</br>
└─ [动物图像识别与分类.pdf](./动物图像识别与分类.pdf)</br>
