# 计划日程

cat-recognition</br>
├─ 3355_130_weights.h5</br>
├─ [README.md](./readme.md) //README文件</br>
├─ SIFT.ipynb</br>
├─ binary</br>
│	├─ .ipynb_checkpoints</br>
│	├─ Classification (cats & dogs).ipynb</br>
│	├─ [Dog Detection (BN).ipynb](./binary/Dog&#32;Detection&#32;(BN).ipynb) //批正则化</br>
│	├─ [Dog Detection (Dropout) .ipynb](./binary/Dog&#32;Detection&#32;(Dropout)&#32;.ipynb) //尝试不同的Dropout</br>
│	├─ [Dog Detection (other kernel size).ipynb](./binary/) //二分类， 不同的卷积核</br>
│	├─ Try_cat_dog.ipynb</br>
│	├─ binary_model.png</br>
│	├─ bn.png</br>
│	├─ droup_out_0.8_0.8.png</br>
│	├─ lr0.0005.png</br>
│	├─ lr0.002.png</br>
│	└─ no_dropout.png</br>
├─ dataCleansing.ipynb</br>
├─ [img](./img/) //数据集</br>
│	├─ cat //15种猫分类</br>
│	├─ detection //二分类数据集</br>
│	└─ no_cat //无猫</br>
├─ img_sub</br>
│	└─ cat</br>
├─ multiclass</br>
│	├─ 150_3355kernal.png</br>
│	├─ 3355_30_weights_1.h5</br>
│	├─ 3355_60_weights_1.h5</br>
│	├─ Classification (clothes).ipynb</br>
│	├─ cat_dataset.png</br>
│	├─ [cat_others_multi.ipynb](./multiclass/cat_others_multi.ipynb) //多分类</br>
│	├─ cd-0.25dp.png</br>
│	├─ f_3355_60.30_weights_1.h5</br>
│	├─ f_3355_60.60_weights_1.h5</br>
│	├─ f_3355_60.90_weights_1.h5</br>
│	├─ f_3355_90.png</br>
│	├─ sub_cd_0.25dp.png</br>
│	└─ sub_cd_0.25dp_150.png</br>
├─ [report.md](./report.md) //总结报告</br>
├─ report_en.md</br>
├─ [web-scraping.ipynb](./web-scraping.ipynb) //拉取图片的爬虫</br>
└─ [动物图像识别与分类.pdf](./动物图像识别与分类.pdf)</br>

## Week 1 (HYX+CYY: 9/12-14/12)

### Plan

+ 熟悉语言
  + Tensorflow
+ 采集数据
  + 网络爬虫
  + 不同清晰度
  + 有/无猫猫
+ 计划方案
  + CNN提取特征
  + 其他算法进行对特征分类
+ __目标__: 完成二维分类（有没有猫）

## Week 2 (HYX+CYY: 16/12-21/12)

+ __目标__: 在识别猫的基础上判断猫的品种

## Week 3 (HYX+CYY: 23/12-28/12)

+ 缓冲时间，完成第二周的剩余任务
+ 尝试对于原方案的CNN进行转移学习运用到更广的范围上

## Week 4 (CYY: 30/12-4/1)

+ 继续完成转移学习方案，并在其他的数据上测试

# Schedule

## Week 1 (Yuxuan & Yongyan: 9/12-14/12)

### Plan

+ Get familiar with Tensorflow
+ Data Collection
  + Web scraping
  + Data with various resolution
  + With/Without cats
+ Algorithms
  + CNN for future selection
  + Use other algorithms for classification
  + Alternative Solution: Use CNN direcrtly in image classification
+ __Aim__: Complete Binary Classification

## Week 2 (Yuxuan & Yongyan: 16/12-21/12)

+ __Aim__: Complete multiclass classification based on the binary classification algorithm

## Week 3 (HYX+CYY: 23/12-28/12)

+ Buffer time. Finish unfinished programmes or improve it.
+ Try to apply transfered the trainned model.

## Week 4 (CYY: 30/12-4/1)

+ Continue finishing transfered learning and try to apply it on other dataset (e.g. SIFCAR-10)
