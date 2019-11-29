# tensorflow dataset的使用

## tf.data基础API的使用

* tf.data.Dataset.from_tensor_slices
* repeat, batch, interleave, map, shuffle, list_files 

### 1.  tf.data.Dataset.from_tensor_slices()

**return**: 从内存中构建一个数据集

**参数**:  列表，numpy数组，字典

```python
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))

# for循环遍历操作
for item in dataset:
    print(item)
```

### 2.  repeat,  batch

**repeat(int64)**: 设置遍历数据集的次数

**batch(int64,drop_remainder=False)**:  设置每次遍历数据集的元素个数

* **drop_remainder**:默认为False,遍历完数据集，True:若最后一次遍历不满足batch则放弃
若不设置batch,默认每次取一个元素
若不设置repeat，会将数据集遍历完
若batch在前，先遍历完数据集，再repeat
若repeat在前，相当于直接遍历repeat倍的数据集

```python
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
```

### 3.  interleave

对现有的dataset中的每一元素做处理得到新的结果，interleave会将这些结果合并起来形成一个新的数据集
case:将不同文件的数据结合成一个大的数据


* map_fn:定义做什么变化
* cycle_length:并行处理的数量
* block_length:从上面的结果中取多少个元素出来

```python
dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v), # map_fn
    cycle_length = 5, # cycle_length
    block_length = 5, # block_length
)
for item in dataset2:
    print(item)
```

## 生成csv文件

```python
# 文件夹名为generate_csv，如果不存在这个文件夹，则新建一个这个名字的文件夹
output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
```

# 将数据保存到csv文件中取

```python
def save_to_csv(output_dir, data, name_prefix,
                header=None, n_parts=10):
    # 生成文件名
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []
    
    for file_idx, row_indices in enumerate(
        # 将np.arange(len(data))分为n_parts个部分
        np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        # 写一个文件
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                # 将数据转换成文件并写到文件中取，并转换为csv 
                f.write(",".join(
                    [repr(col) for col in data[row_index]]))
                f.write('\n')
    return filenames
```

# np.c_():将数据按行合并
train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test] 
header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ",".join(header_cols)

train_filenames = save_to_csv(output_dir, train_data, "train",
                              header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, "valid",
                              header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data, "test",
                             header_str, n_parts=10)
```

## tf.dataset读取csv文件

* csv
  * tf.data.TextLineDataset, tf.io.decode_csv

### 1. filename -> dataset

```python
# list_files:将文件名生成一个Dataset
filename_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filename_dataset:
    print(filename)
```

### 2. read file -> dataset -> datasets -> merge

tf.data.TextLineDataset:按行读取文本文件形成dataset

```python
n_readers = 5
dataset = filename_dataset.interleave(
    # skip(1):省略第一行
    lambda filename: tf.data.TextLineDataset(filename).skip(1),
    # 并行读取文件
    cycle_length = n_readers
)
for line in dataset.take(15):
    print(line.numpy())
```

### 3. parse csv

tf.io.decode_csv(str, record_defaults):解析csv文件

```python
sample_str = '1, 2, 3, 4, 5'
# record_defaults:定义默认值和类型
record_defaults = [tf.constant(0, dtype = tf.int32)] * 5
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fields)
```


### tf.io.decode_csv的使用:解析csv

参数：
str:要操作的元素
record_defaults:定义类型和默认值，长度应与str相同

```python
sample_str = '1, 2, 3, 4, 5'
# 默认值为0，类型为int32
record_defaults = [tf.constant(0, dtype = tf.int32)] * 5
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fields)
```

**将数据中的一行转换出来**

```pyhotn
def parse_csv_line(line, n_fields = 9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    # tf.stack(a):将a转换为向量
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

parse_csv_line(b'-0.9868720801669367,0.832863080552588,-0.18684708416901633,-0.14888949288707784,-0.4532302419670616,-0.11504995754593579,1.6730974284189664,-0.7465496877362412,1.138',
               n_fields=9)

```

**转换整个数据集**

```python
# 1. filename -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv
# batch_size:读取多少个数据生成数据集
def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    # repeat不加参数，将这个数据集重复无限次
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length = n_readers
    )
    # shuffle:将数据进行混排
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

train_set = csv_reader_dataset(train_filenames, batch_size=3)
#.take(2):取前两行
for x_batch, y_batch in train_set.take(2):
    print("x:")
    pprint.pprint(x_batch)
    print("y:")
    pprint.pprint(y_batch)
```

**建立模型**

```python

batch_size = 32
train_set = csv_reader_dataset(train_filenames,
                               batch_size = batch_size)
valid_set = csv_reader_dataset(valid_filenames,
                               batch_size = batch_size)
test_set = csv_reader_dataset(test_filenames,
                              batch_size = batch_size)
 
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                       input_shape=[8]),
    keras.layers.Dense(1),
])
model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]

history = model.fit(train_set,
                    validation_data = valid_set,
                    steps_per_epoch = 11160 // batch_size,
                    validation_steps = 3870 // batch_size,
                    epochs = 100,
                    callbacks = callbacks)

model.evaluate(test_set, steps = 5160 // batch_size)
```


## Dataset读取tfrecord(tensorflow独有的数据格式，对其有一定的优化)

* tfrecord
  * tf.train.FloatList, tf.train.Int64List, tf.train.BytesList
  * tf.train.Feature, tf.train.Features, tf.train.Example
  * example.SerializeToString
  * tf.io.ParseSingleExample
  * tf.io.VarlenFeature, tf.io.FixedLenFeature
  * tf.data.TFRecordDataset, tf.io.TFRecordOptions·

### tfrecord基础API的使用

* tfrecord 文件格式
  * tf.train.Example--可以是一个样本或一组样本
    * tf.train.Features -> {'key': tf.train.Feature}
      * key为feature的名字，value为具体的值
      * 不同的Feature有不同的格式
        * tf.train.ByteList/FloatList/Int64List

```python

# name.encode('utf-8'):将name转换为utf-8的格式
favorite_books = [name.encode('utf-8')
                  for name in ["machine learning", "cc150"]]
# 获得BytesList的对象
favorite_books_bytelist = tf.train.BytesList(value = favorite_books)
print(favorite_books_bytelist)

hours_floatlist = tf.train.FloatList(value = [15.5, 9.5, 7.0, 8.0])
print(hours_floatlist)

age_int64list = tf.train.Int64List(value = [42])
print(age_int64list)

# 定义features
features = tf.train.Features(
    feature = {
        "favorite_books": tf.train.Feature(
            bytes_list = favorite_books_bytelist),
        "hours": tf.train.Feature(
            float_list = hours_floatlist),
        "age": tf.train.Feature(int64_list = age_int64list),
    }
)
print(features)
```

生成example

```python
example = tf.train.Example(features=features)
print(example)
# 将example序列化--目的是对文件进行压缩
serialized_example = example.SerializeToString()
print(serialized_example)
```

将example存到tfrecord中去，生成一个tfrecord文件

```python
output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = "test.tfrecords"
filename_fullpath = os.path.join(output_dir, filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)

```

读取tfrecord文件

```python
expected_features = {
    # VarLenFeature:变长的feature
    # FixedLenFeature:定长的Feature
    "favorite_books": tf.io.VarLenFeature(dtype = tf.string),
    "hours": tf.io.VarLenFeature(dtype = tf.float32),
    "age": tf.io.FixedLenFeature([], dtype = tf.int64),
}
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features)
    # sparse_tensor:存储稀疏矩阵时效率较高
    books = tf.sparse.to_dense(example["favorite_books"],
                               default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))
```

将文件存储为压缩文件

```python
filename_fullpath_zip = filename_fullpath + '.zip'
options = tf.io.TFRecordOptions(compression_type = "GZIP")
with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
    for i in range(3):
        writer.write(serialized_example)
```

读取压缩后的文件

```python
dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], 
                                      compression_type= "GZIP")
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features)
    books = tf.sparse.to_dense(example["favorite_books"],
                               default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))
```

### 生成tfrecord文件

获取文件名

```python

source_dir = "./generate_csv/"
# print(source_dir) 
def get_filenames_by_prefix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir, filename))
    return results

train_filenames = get_filenames_by_prefix(source_dir, "train")
valid_filenames = get_filenames_by_prefix(source_dir, "valid")
test_filenames = get_filenames_by_prefix(source_dir, "test")

import pprint
pprint.pprint(train_filenames)
pprint.pprint(valid_filenames)
pprint.pprint(test_filenames)

```

从csv文件中读取训练集，验证集，测试集

```python

def parse_csv_line(line, n_fields = 9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length = n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

batch_size = 32
train_set = csv_reader_dataset(train_filenames,
                               batch_size = batch_size)
valid_set = csv_reader_dataset(valid_filenames,
                               batch_size = batch_size)
test_set = csv_reader_dataset(test_filenames,
                              batch_size = batch_size)

```


```python

def serialize_example(x, y):
    """Converts x, y to tf.train.Example and serialize"""
    input_feautres = tf.train.FloatList(value = x)
    label = tf.train.FloatList(value = y)
    features = tf.train.Features(
        feature = {
            "input_features": tf.train.Feature(
                float_list = input_feautres),
            "label": tf.train.Feature(float_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

# compression_type:压缩方法
def csv_dataset_to_tfrecords(base_filename, dataset,
                             n_shards, steps_per_shard,
                             compression_type = None):
    options = tf.io.TFRecordOptions(
        compression_type = compression_type)
    all_filenames = [] # 保存文件名
    for shard_id in range(n_shards):
        filename_fullpath = '{}_{:05d}-of-{:05d}'.format(
            base_filename, shard_id, n_shards)
        with tf.io.TFRecordWriter(filename_fullpath, options) as writer:
            for x_batch, y_batch in dataset.take(steps_per_shard):
                # 解开batch
                for x_example, y_example in zip(x_batch, y_batch):
                    writer.write(
                        serialize_example(x_example, y_example))
        all_filenames.append(filename_fullpath)
    return all_filenames
```