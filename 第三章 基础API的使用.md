# 第三章 基础API的介绍和使用:+1:

## 基础数据类型
  * tf.constant, tf.string
  * tf.ragged.constant, tf.SparseTensor, tf.Variable
* 自定义损失函数——tf.reduce_mean
* 自定义层次——keras.layers.Lambda和继承法
* tf.function
  * tf.function, tf.autograph.to_code, get_concrete_function
* GraphDef
  * get_operations, get_operation_by_name
  * get_tensor_by_name, as_graph_def
* 自动求导
  * tf.GradientTape
  * Optimzier.apply_gradients

### 基础API与keras的集成:pig:

- 自定义损失函数

- 自定义层次

1. 

```python
    layer = tf.keras.layers.Dense(100)
    # input_shape:输入维度(数据的size)
    layer = tf.keras.layers.Dense(100, input_shape(None, 5)) # 输出为100 * None的矩阵
    layer(tf.zeros([10,5])) # 定义一个10*5的0矩阵为layer的输入，则输出为100*10的矩阵
```

layer.variables: 打印layer中的所有参数
layer.trainable_variabes: 打印可训练的变量
help(layer): 查看layer中的其他方法

2. 

```python
    # customized dense layer.
    class CustomizedDenseLayer(keras.layers.Layer): # 继承自keras.layers
        def __init__(self, units, activation=None, **kwargs):
            self.units = units
            self.activation = keras.layers.Activation(activation)
            # 调用父类的函数
            super(CustomizedDenseLayer, self).__init__(**kwargs)
    
        def build(self, input_shape):
            """构建所需要的参数"""
            # x * w + b. input_shape:[None, a] w:[a,b]output_shape: [None, b]
            # add_weight是父类Layer中的一个子方法：用来得到一个变量
            # shape:矩阵的维度
            # initializer:如何随机初始化参数，'uniform':均匀分布随机化
            # trainable = True：参数可被训练
            self.kernel = self.add_weight(name = 'kernel',
                                      shape = (input_shape[1], self.units),
                                      initializer = 'uniform',
                                      trainable = True)
            self.bias = self.add_weight(name = 'bias',
                                    shape = (self.units, ),
                                    initializer = 'zeros',
                                    trainable = True)
            # 调用父类的函数
            super(CustomizedDenseLayer, self).build(input_shape)
    
        def call(self, x):  # 完成一次正向运算，如何从输入到输出
            """完成正向计算"""
            return self.activation(x @ self.kernel + self.bias) 
```

* 自定义激活函数

```python
# tf.nn.softplus: log(1 + e^x)
customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))
print(customized_softplus([-10., -5., 0., 5., 10.]))

model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu',
                         input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # customized_softplus等价于下面两个
    # keras.layers.Dense(1, activation="softplus"),
    # keras.layers.Dense(1), keras.layers.Activation('softplus'),
])
```


### @tf.function的使用:weary:

- 图结构
- 将python函数编译成tensorflow的图
- 易于将模型导出称为GraphDef+checkpoint或者SavedModel
- 使得eager execution可以默认打开(如果没有@tf.function,虽然可以用eager execution写代码，但是不能保存模型结果)
- 1.0的代码可以通过tf.function来继续 在2.0里使用
- 替代session

##### tf.function and autograph:tired_face:

* tf.function: 将普通的python函数和代码块转换为tensorflow中的图
* autograph: tf.function依赖的机制，转换中的机制

##### 将普通的python函数转换为tensorflow中的图

1. g(tensorflow) = tf.function(f(python)) 

```python
def scaled_elu(z, scale = 1.0, alpha = 1.0):
    # z >= 0 ? scale * z: scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., -2.5])))
# 将scaled_elu转换为tensorflow的图结构———速度实现上tensorflow实现比单纯的python函数要快--GPU上差距更明显
scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))
# 找回python函数
print(scaled_elu_tf.python_function is scaled_elu)
```

2. @tf.function

```python
# 1 + 1/2 + 1/2^2 + ... + 1/2^n

@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total

print(converge_to_2(20))
```

##### 注意事项

1.在定义variable时要放到@tf.function外面，否则会报错

```python
var = tf.Variable(0.)

@tf.function
def add_21():
    return var.assign_add(21) # += 

print(add_21())
```

#### 使用@tf.function给函数的输入添加类型限定

```python
# input_signature: 定义函数的输入类型
# 定义输入类型为int32
@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)

try:
    print(cube(tf.constant([1., 2., 3.])))
except ValueError as ex:
    print(ex)
    
print(cube(tf.constant([1, 2, 3])))
```


**记录：保存模型和加载模型**

@tf.function 可以将一个普通的python函数转化为tensorflow中的图

get_concrete_function可以给通过@tf.function转化的python函数加上一个input signature，从而让这个python函数成为一个可以保存的图结构

```python

# @tf.function py func -> tf graph
# get_concrete_function -> add input signature -> SavedModel

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32))
print(cube_func_int32)
```

是否有相同的signature

```python
print(cube_func_int32 is cube.get_concrete_function(
    tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(
    tf.constant([1, 2, 3])))
# 取出图
cube_func_int32.graph
# 差看图中的操作
cube_func_int32.graph.get_operations()
# 打印操作信息
pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)
# 打印操作的输入输出信息
print(list(pow_op.inputs))
print(list(pow_op.outputs))
# 通过名字得到相应的操作
cube_func_int32.graph.get_operation_by_name("x")
# 通过名字获得tensor
cube_func_int32.graph.get_tensor_by_name("x:0") # 一般后面加个':0'
# 获得图定义
cube_func_int32.graph.as_graph_def()
```


### 自定义求导

初高中方法:

1.f(x)

```python

def f(x):
    return 3. * x ** 2 + 2. * x - 1

def approximate_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)

print(approximate_derivative(f, 1.))
```

2.g(x,y)

```python
def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)

def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2

print(approximate_gradient(g, 2., 3.))
```

使用tf.GradientTape求导

```python
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
# 默认persistent = False，运行一次后关闭，True需要手动关闭
with tf.GradientTape(persistent = True) as tape:
    # 定义需要求导的函数
    z = g(x1, x2) # z = (x1 + 5) * (x2 ** 2)

dz_x1 = tape.gradient(z, x1)
dz_x2 = tape.gradient(z, x2)
print(dz_x1, dz_x2)

del tape # 关闭GradientTape
# 也可以运行一次
'''
dz_x1x2 = tape.gradient(z, [x1, x2])

print(dz_x1x2)
'''
```

同时对两个函数求导

```python
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x ** 2
tape.gradient([z1, z2], x) # 输出会将两个函数求导结果相加
```

关注constant的导数

```python
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    # 告诉GradientTape需要关注导数的constant
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])

print(dz_x1x2)
```

求二阶导数--使用嵌套

```python

x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2])
               for inner_grad in inner_grads]
print(outer_grads)
del inner_tape
del outer_tape
```

更新参数-tf.GradientTape实现

```python
learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx) # x -= learning_rate * da_dx
print(x)
```

更新参数-Optimzier.apply_gradients实现

```python
learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr = learning_rate)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dz_dx, x)])
print(x)
```
