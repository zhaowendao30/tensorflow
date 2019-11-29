 
# 第二章 Tensorflow-keras实战



## 理论部分


* :one: tensorflow-keras简介
* :two: 分类问题、回归问题、损失函数
* :three: 神经网络、激活函数、批归一化、Dropout
* :four: Wide & deep 模型
* :five: 超参数搜索

## 实战部分

* Keras搭建分类模型
* Keras回调函数
* Keras搭建回归模型
* Keras搭建深度神经网络
* Keras实现 wide & deep 模型
* Keras与scikit-learn实现超参数搜索

### keras是什什么

* 基于python的高级神经网络API
* Francois Chollet与2014-2015编写Keras
* 以Tensorflow、CNTK或者Theano为后端运行，keras必须有后端才可以进行
* 后端可以切换，现在多用tensorflow
* 极方便于快速实验，帮助用户以最少的时间验证自己的想法

### tensortflow-keras是什么

* tensorflow对keras API规范的实现
* 相对于tensorflow为后端的keras，tensorflow-keras与tensorflow结合更加紧密
* 实现在tf.keras空间下
  
### tf-keras和keras联系

* 基于同一套API
  * keras程序可以通过改导入方式轻松转为tf.keras程序
  * 反之可能不成立，因为tf.keras有其他特性
* 相同JSON和HDF5模型序列化格式和语义

### tf-keras 和 keras区别

* tf.keras全面支持eager mode
* 只是用keras.Sequential和keras.Model时没影响
* 自定义Model内部运算逻辑的时候会有影响
* tf底层API可以使用keras的model.fit等抽象
* 适用于研究人员
* tf.keras支持基于tf.data的模型训练
* tf.keras支持TPU训练
* tf.keras支持tf.distribution中的分布式策略
* tf.keras可以与tensorflow中的estimator集成
* tf.keras可以保存为SavedModel(可以在其他平台运行)

### 如何选择keras和tf.keras

* 如果想用tf.keras的任何一个特性，那么选tf.keras
* 如果后端互换性很重要，那么选keras
* 如果都不重要，随便

### 目标函数

分类问题

* 需要衡量目标类别与当前预测的差距
  * 三分类问题输出例子: [0.2, 0.7, 0.1]
  * 三分类真实类别: 2 -> one_hot -> [0, 0, 1]

* one-hot编码，把正整数变为向量表达
  * 生成一个长度不小于整整数的向量，只有正整数的位置处为1，其余位置都为0

### 损失函数

* 平方差损失
* 交叉熵损失


************************

## 分类实战--神经网络

### 分类神经网络步骤

#### 1.创建Sequential模型

```python
model = keras.models.Sequential()
```

#### 2.添加网络层，定义神经元的数目和该层的激活函数

```python
# 神经元数目为100，激活函数为relu
# relu = max(0,x)
# 第一层输入层要定义输入数据的维度
model.add(kerass.layer.Flatten(input_shape = [28, 28]))
model.add(keras.layer.Dense(100, activation = 'relu'))
```

#### 3.使用complie方法定义网络结构

**complie方法的参数及解释:**

* optimizer：模型调整参数的方法。从 tf.train 模块向其传递优化器实例
  * eg: AdamOptimizer、RMSPropOptimizer 或 GradientDescentOptimizer。
* loss：损失函数。损失函数由名称或通过从 tf.keras.losses 模块传递可调用对象来指定。
  * rg: 均方误差 (mse)、categorical_crossentropy 和 binary_crossentropy。
* metrics：用于监控训练。它们是 tf.keras.metrics 模块中的字符串名称或可调用对象。


```python
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

#### 4.使用fit方法拟合模型和数据

**fit方法的参数及解释：**

* 训练集数据:x_train, 对应的分类标签:y_train
* 训练批次:epochs(遍历训练集的次数)
* 验证集数据:validation = (x_valid, y_valid)
* bacth_size:模型将数据分成较小的批次，并在训练过程中迭代这些批次，batch_size为批次的大小(可以不用)
* callbacks: 回调函数
  * EarlyStopping: 比如当训练模型时loss不在下降时提前结束训练
    * monitor: 表示所关注的指标，一般默认为损失函数的值
    * min_data: 表示为变化的下限
    * patience： 表示为连续低于下限的最大次数
    * 若损失函数连续p次(p>patient)变化的值小于min_data，则模型提前结束训练。
  * ModelCheckpoint: 每隔一段时间保存模型的参数
  * TensorBoard: 在训练模型的过程中，实时可视化模型参数以及loss，accuracy的变化。

```python
logdir = './callbacks'
# 如果没有callbacks文件夹，就创建一个callbacks文件夹
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                 "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only = True),    # 将最好的模型储存在callbacks文件中
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]


history = model.fit(x_train_scaled, y_train, epochs=10,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks = callbacks)
```
该方法会返回训练过程中的一些结果

#### 5.评估和预测

```python
# x , y为分别的样本与对所对应的标签，
# dataset为打包好的数据集，包含样本特征与标签
 
model.evaluate(x, y, batch_size=32) # or model.evaluate(dataset, steps=30)
 
model.predict(x, batch_size=32) # or model.predict(dataset, steps=30)
```

### 归一化--输入数据进行归一化

* Min-max归一化: $x * =(x-min)/(max-min)$
* Z-score归一化: $x * = (x-u)/std$

```python
# x = (x - u) / std

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train: [None, 28, 28] -> [None, 784]
# transform为归一化函数，接受的参数为二维矩阵，我们的输入为三维的，先通过reshape(-1,1)将其变为二维的(如上所示)，然后reshape(-1,28,28)将其变回三维的
# fit的功能是将均值和方差记录下来，训练集，验证集和测试集的所有数据归一化所用的均值-方差都是训练集的
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
# 验证集不用fit
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
```

### 批归一化--对每层的激活值做归一化

他可分为在激活之前做归一化和激活之后做归一化

```python
# tf.keras.models.Sequential()
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    # 批归一化在激活函数之后，即上一层输出值先激活，然后再做归一化
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    """
    # 批归一化在激活函数之前，即上一层输出值先进行归一化处理然后再激活
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    """
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
```

### Dropout的作用

* 防止过拟合
* 训练集上很好，测试集上不好
* 参数太多，记住样本，不能泛化

```python
# tf.keras.models.Sequential()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    # selu是一个自带归一化的激活函数
    model.add(keras.layers.Dense(100, activation="selu"))
model.add(keras.layers.AlphaDropout(rate=0.5))
# AlphaDropout: 1. 均值和方差不变--分布不变 2. 归一化性质也不变
# rate为丢失神经元的概率
# model.add(keras.layers.Dropout(rate=0.5)) -- 纯净Dropout--可能是分布发生改变
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
```


****


## 回归实战

```python
model = keras.models.Sequential(
    keras.layer.Flatten(30, input_shape = x_train.shape[1:])
    keras.layer.Dense(1)
)
model.complie(loss = 'mean_squared_error'
              optimizer = ['sgd']  
)
callbacks = [keras.callbacks.EarlyStopping(patient = 5, min_delta = 1e-2)]
history = model.fit(x_train, y_train,
                    epochs =  100,
                    validation = (x_valid, y_valid),
                    callbacks = callbacks)
```

### 超参数--神经网络训练过程中不变的参数

* 网络结构参数：几层，每层宽度，每层激活函数等
* 训练参数：batch_size, 学习率，学习率衰减算法等
* 手工去试耗费人力

#### 搜索策略

* 网格搜索
定义n维方格：每个方格对应一组超参数，一组一组参数尝试
* 随机搜索
参数的生成方式随机，可探索的空间更大
* 遗传算法搜索
  * A.初始化候选参数几何->训练->得到模型指标作为生存概率
  * B.选择(选择一些参数集合)->交叉(不同参数集合中各选取一部分)->变异(对参数集合中某一个参数或者某几个参数进行微小的调整)->产生下一代集合 
  * C.重新到A
* 启发式搜索(热点)--研究热点-**AutoML**：神经网络结构搜索
  * 使用循环神经网络来生成参数
  * 使用强化学习来进行反馈，使用模型来训练生成参数

**1.for循环**

```python
# learning_rate: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
# W = W + grad * learning_rate
# for循环不能并行化，现实中参数很多，需要多层循环
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
histories = []
for lr in learning_rates:
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu',
                           input_shape=x_train.shape[1:]),
        keras.layers.Dense(1),
    ])
    optimizer = keras.optimizers.SGD(lr)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    callbacks = [keras.callbacks.EarlyStopping(
        patience=5, min_delta=1e-2)]
    history = model.fit(x_train_scaled, y_train,
                        validation_data = (x_valid_scaled, y_valid),
                        epochs = 100,
                        callbacks = callbacks)
    histories.append(history)
```

**2.sklearn**

* 1.将tf.keras的model转化为sklearn的model
* 先定义好一个tf.keras的model，然后调用 一个函数来把tf.keras的model封装成sklearn的model-
* 2.定义参数集合
* 3.搜索参数

```python
def build_model(hidden_layers = 1,
                layer_size = 30,
                learning_rate = 3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu',
                                 input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size,
                                     activation = 'relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss = 'mse', optimizer = optimizer) # 'mse'='mean_squared_error'
    return model

sklearn_model = KerasRegressor(
    build_fn = build_model)
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = sklearn_model.fit(x_train_scaled, y_train,
                            epochs = 10,
                            validation_data = (x_valid_scaled, y_valid),
                            callbacks = callbacks)
```

```python
from scipy.stats import reciprocal # reciprocal为一个分布
# reciprocal分布函数 f(x) = 1/(x*log(b/a)) a <= x <= b

param_distribution = {
    "hidden_layers":[1, 2, 3, 4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal(1e-4, 1e-2),
}

from sklearn.model_selection import RandomizedSearchCV
# 初始化搜索对象
# n_iter参数集合
# n_jobs有多少个任务在并行处理
random_search_cv = RandomizedSearchCV(sklearn_model,
                                      param_distribution,
                                      n_iter = 10,
                                      cv = 3,
                                      n_jobs = 1)
random_search_cv.fit(x_train_scaled, y_train, epochs = 100,
                     validation_data = (x_valid_scaled, y_valid),
                     callbacks = callbacks)

# cross_validation: 训练集分成n份，n-1训练，最后一份验证. 默认为3，参数为cv
```

```python
# 搜索空间
from scipy.stats import reciprocal
reciprocal.rvs(1e-4, 1e-2, size = 10)
print(random_search_cv.best_params_) # 最好的参数
print(random_search_cv.best_score_)  # 最好的分值
print(random_search_cv.best_estimator_) # 最好的model
model = random_search_cv.best_estimator_.model # 获取最好的model
model.evaluate(x_test_scaled, y_test) # 在测试集上应用最好的model
```


### 稀疏特征--优缺点

* 离散值特征
* One-hot表示
* Eg: 专业={计算机， 人文， 其他}. 人文 = [0, 1, 0]
* Eg:词表={人工智能，你，他，慕课网，...}.他=[0,0,1,0,...]
* 叉乘={(计算机，人工智能)，(计算机，你),...}
  * 叉乘可以用来刻画一个样本
  * 叉乘之后
* 稀疏特征做叉乘获取共现信息
* 实现记忆的效果

#### 优点

* 有效，广泛用于工业界 例如：广告点击率预估、推荐算法

#### 缺点

* 需要人工设计: 现实中的物体有太多的维度，需要人工选取叉乘的特征
* 可能过拟合，所有特征都叉乘，相当于记住每一个样本

### 密集特征--优缺点

* 向量表达
eg: 词表={人工智能，你，他，慕课网}.
    他={0.3,0.2,0.6,(n维向量)}
* Word2vec工具
男-女=国王-王后


#### 优点

* 带有语义信息 ，不同向量之间有相关性
* 兼容没有出现过的特征组合
* 更少人工参与


#### 缺点

* 过度泛化，推荐不怎么相关的产品


