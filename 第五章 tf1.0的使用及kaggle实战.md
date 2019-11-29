# 第五章 tf1.0的使用及kaggle实战

## tf.estimator使用

* keras转estimator
将keras搭建的模型转为estimator
* 使用预定义的estimator
  * BaseLineClassifier:机器模型，通过随机参数的方式进行预测
  * LinearClassifier:线性模型
  * DNNClassifier:深度全连接神经网络
* tf.feature_column做特征工程:将数据表达成feature，利用feature做一些操作

## API列表

* tf.keras.estimator.to_estimator:将keras模型转为estimator
  * train, evaluate
* tf.estimator.BaselineClassifier
* tf.estimator.LinearClassifier
* tf.estimator.DNNClassifier
* tf.feature_column
  * categorical_column_with_vovabulary_list
  * numeric_column
  * indicator_column
  * cross_column
* keras.layers.DenseFeatures:将feature_column表达的这种数据表达到网络中去


tf.feature_column.categorical_column_with_vocabulary_list:定义了一个feature——column
tf.feature_column.indicator_column:对离散特征进行one_hot编码