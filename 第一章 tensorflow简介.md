# 第一章 tensorflow 简介和环境搭建

## tensorflow 是什么、tensorflow历史、tensorflow vs pytorch

### 数据流图

* 节点——处理数据
* 线——节点间的输入输出关系
* 线上运输张量
* 节点被分配到各种计算设备上运行

### 特性

* 高度灵活性
* 真正的可移植性——可以在各个设备上运行
* 产品和科研结合
* 自动求微分
* 多语言支持
* 性能最优化

### tensorflow1.0——主要特性

* XLA——Accelerate Linear Algebra
  * 提升训练速度58倍
  * 可以再移动设备上运行
* 引入更高级的API——tf.layers/tf.metrics/tf.losses/tf.keras
* tensorflow调试器
* 支持docker镜像，引入tensorflow serving服务

### tensorflow2.0——主要特性

* 使用tf.keras和eager model进行更简单的模型构建
* 鲁棒的跨平台模型部署
* 强大的研究实验
* 清除不推荐使用的API和减少重复来简化API

### tensorflow2.0——简化的模型开发流程

* 使用tf.data加载数据
* 使用tf.keras构建模型，也可以用premade estimator来验证模型
  * 使用tensorflow hub进行迁移学习
* 使用eager model进行运行和调试
* 导出到SavedModel
* 使用tensorflow serve、tensorflow lite、tensorflow.js部署模型

### tensorflow2.0——强大的跨平台功能

* tensorflow服务
  * 之间通过HTTP/REST或GRPC/协议缓冲区
* tensorflow lite——可部署在Android、IOS和嵌入式系统上
* tensorflow.js——在javascript中部署模型
* 其他语言

### tensorflow2.0——强大的研究实验

* Keras功能API和子类API,允许创建括扑结构
* 自定义训练逻辑，使用tf.GradientTape和tf.custom_gradient进行更细粒度的控制
* 底层API自始至终可以与高层结合使用，完全的可定制
* 高级扩展：Ragged Tensors、Tensor2Tensor等

### tensorflow VS pytorch

* 入门时间
  * tensorflow1.x
    * 静态图-效率高
  * 学习额外概念
    * 图、会话、变量、占位符等
    * 写样板代码
  * tensorflow2.0
    * 动态图—调试容易
    * Eager mode避免1.0缺点，直接集成在python中
  * pytorch
    * 动态图
    * Numpy的扩展，直接集成在python中
* 图创建和调试
* 全面性
* 序列化和部署

***

**实现1 + 1/2 + 1/2^2 + 1/2^3 + ... +1/2^50**

* **tensroflow 1.x**

```python
import tensorflow as tf
print(tf.__version__)

x = tf.Variable(0.)
y = tf.Variable(1.)
print(x)
print(y)
# x = x + y
add_op = x.assign(x + y)
# y = y / 2
diy_op = y.assgin(y / 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(50):
        sess.run(add_op)
        sess.run(div_op)
    print(x.eval()) # sess.eval(x)
```

* **pytorch**

```python
print(torch.__version__)

x = torch.Tensor([0.])
y = torch.Tensor([1.])
for iteration in range(50):
    x = x + y
    y = y / 2
print(x)
```

* **tensorflow 2.0**

```python
import tensorflow as tf
#tf.enable_eager_execution()

print(tf.__version__)

x = tf.constant(0.)
y = tf.constant(1.)
for iteration in range(50):
    x = x + y
    y = y / 2
print(x.numpy())
```

* **python**

```python
x = 0
y = 1
for iteration in range(50):
    x = x + y
    y = y / 2
print(x)
```

### 图创建和调试

* tensorflow 1.x
  * 静态图，难以调试，学习tfdbg调试
* tensorflow 2.0 与Pytorch
  * 动态图，python自带调试工具 

#### 全面性

* pytorch缺少
  * 沿维翻转张量(np.flip, np.flipud, np.fliplr)
  * 检查无穷与非数值张量(np.is_nan, np.is_inf)
  * 快速傅里叶变换(np.fft)
* 随时间变换，越来越接近
  
#### 序列化与部署

* tensorflow支持更加广泛
  * 图保存为protocol buffer
  * 跨语言
  * 跨平台
* pytorch支持更简单

### 环境配置

本地配置

* [Virtualenv安装](www.tensorflow.org/install/pip)
* [GPU版环境配置](https://blog.csdn.net/u014595019/article/details/53732015)
  *
  * 安装显卡驱动- > Cuda安装- > Cudnn安装
  * tensorflow安装

云端配置

* 为什么要在云上配置
  * 规格统一，节省自己的机器
  * 有直接配置好环境的镜像

* 云环境
  * Google Cloud配置——怂300刀免费体验
  * Amazon云配置