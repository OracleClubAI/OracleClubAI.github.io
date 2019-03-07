---
layout: post
title:  "将Keras作为tensorflow的精简接口"
categories: tensorflow
tags:  Keras tensorflow 
author: 陈权
---

* content
{:toc}

>本文地址：[https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)

>本文作者：Francois Chollet

![](/img/picture/keras-tensorflow-logo.jpg)


在tensorflow中调用Keras层
让我们以一个简单的例子开始：MNIST数字分类。

我们将以Keras的全连接层堆叠构造一个TensorFlow的分类器，

```python
import tensorflow as tf
sess = tf.Session() # 建立TensorFlow回话机制

from keras import backend as K
# 将已建立的回话机制传给Keras调用
K.set_session(sess) 
```

然后，我们开始用tensorflow构建模型：

```python
# 定义网络输入数据的占位符
inputs = tf.placeholder(tf.float32, shape=(None, 784))
outputs = labels = tf.placeholder(tf.float32, shape=(None, 10))
```
用Keras可以加速模型的定义过程：
```python
from keras.layers import Dense

# Keras 层可以调用TensorFlow的Tensor
x = Dense(128, activation='relu')(inputs)  # 使用relu激活函数的128个神经单元的全连接层
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # 输入层使用softmax激活函数，10个神经单元（对应分类数）

```

定义损失函数：
```python
from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(outputs, preds))
```

然后，我们可以用tensorflow的优化器来训练模型：
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.AdmaOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})
```
模型评估：
```python
from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(outputs, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels})
```

我们只是将Keras作为生成从tensor到tensor的函数（op）的快捷方法而已，优化过程完全采用的原生tensorflow的优化器，而不是Keras优化器，因此我们压根不需要Keras的Model

**注意：虽然有点反直觉，但Keras的优化器要比TensorFlow的优化器快大概`5-10%`。虽然这种速度的差异基本上没什么差别。**

**训练和测试行为不同**

有些Keras层，如`BN`，`Dropout`，在训练和测试过程中的行为不一致，你可以通过打印`layer.uses_learning_phase`来确定当前层工作在训练模式还是测试模式。

如果你的模型包含这样的层，你需要指定你希望模型工作在什么模式下，通过Keras的`backend`你可以了解当前的工作模式：
```python
from keras import backend as K
print(K.learning_phase())
```

在使用回话机制的过程中，向`feed_dict`中传递`1（训练模式）或0（测试模式）`即可指定当前工作模式：
```python
# 训练模型
train_step.run(feed_dict={x: batch[0], labels: batch[1], K.learning_phase(): 1})

# 测试模型
acc_value.eval(feed_dict={x: batch[0], labels: batch[1], K.learning_phase(): 0})

```

一个训练和测试行为不同的例子：

```python
from keras.layers import Dropout,Dense
from keras import backend as K
import tensorflow as tf

inputs = tf.placeholder(tf.float32, shape=(None, 784))
outputs = tf.placeholder(tf.float32, shape=(None, 10))

x = Dense(128, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1],
                                  K.learning_phase(): 1})

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0})
```

**Keras与TensorFlowd的变量名作用域和设备作用域的兼容**

Keras的层与模型和tensorflow的命名完全兼容，例如：

```python
from keras.layers import LSTM

x = tf.placeholder(tf.float32, shape=(None, 20, 64))
with tf.name_scope('block1'):
    y = LSTM(32, name='mylstm')(x)
```
我们LSTM层的权重将会被命名为block1/mylstm_W_i, block1/mylstm_U等

类似的，设备的命名也会像你期望的一样工作：
```python
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)
```
所有的计算和变量将会在`gpu:0`运行。

**Keras与TensorFlowd的变量作用域兼容和Graph的作用域兼容**

1、变量作用域兼容

变量共享应通过多次调用同样的Keras层或模型来实现，而不是通过TensorFlow的变量作用域实现。
TensorFlow变量作用域将对Keras层或模型没有任何影响。
更多Keras权重共享的信息请参考[这里](https://keras.io/zh/getting-started/functional-api-guide/#_5)

Keras通过重用相同层或模型的对象来完成权值共享，这是一个例子：
```python

from keras.layers import LSTM
# 定义一个Keras的LSTM层
lstm = LSTM(32)

# 定义两个占位符
x = tf.placeholder(tf.float32, shape=(None, 20, 64))
y = tf.placeholder(tf.float32, shape=(None, 20, 64))

# 这两个LSTM网络的权值是共享的
x_encoded = lstm(x)
y_encoded = lstm(y)

```

2 、Graph的作用域兼容

任何在tensorflow的Graph作用域定义的Keras层或模型的所有变量和操作将被生成为该Graph的一个部分，
例如，下面的代码将会以你所期望的形式工作：

```python
from keras.layers import LSTM
import tensorflow as tf

my_graph = tf.Graph()
with my_graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)      # all ops / variables in the LSTM layer are created as part of our graph
```
LSTM中所有的计算和变量将会在`my_graph`中被创建。

**收集可训练权重与状态更新**

某些Keras层，如状态`RNN`和`BN`层，其内部的更新需要作为训练过程的一步来进行，
这些更新被存储在一个tensor tuple里：`layer.updates`，
你应该生成assign操作来使在训练的每一步这些更新能够被运行，这里是例子：

```python
from keras.layers import BatchNormalization

layer = BatchNormalization()(x)

update_ops = []
for old_value, new_value in layer.updates:
    update_ops.append(tf.assign(old_value, new_value))
```
注意如果你使用Keras模型，`model.updates`将与上面的代码作用相同（收集模型中所有更新）

另外，如果你需要显式的收集一个层的可训练权重，你可以通过`layer.trainable_weights`来实现，
对模型而言是`model.trainable_weights`，它是一个tensorflow变量对象的列表：
```python
from keras.layers import Dense

layer = Dense(32)(x)  # 定义和调用一个层
print (layer.trainable_weights)  # 获取当前TensorFlow中当前层的所有变量
```

### 使用Keras模型与TensorFlow协作

**将Keras Sequential模型转换到TensorFlow中**

假如你已经有一个训练好的Keras模型，如VGG-16，现在你想将它应用在你的TensorFlow工作中，应该怎么办？

首先，注意如果你的预训练权重含有使用`Theano`训练的卷积层的话，你需要对这些权重的卷积核进行转换，
这是因为`Theano`和`TensorFlow`对卷积的实现不同(Theano为`channels_first`,TensorFlow为`channels_last`)，`TensorFlow`和`Caffe`实际上实现的是相关性计算。
点击[这里](https://github.com/fchollet/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa)查看详细示例。

假设你从下面的Keras模型开始，并希望对其进行修改以使得它可以以一个特定的tensorflow张量`my_input_tensor`
为输入，这个tensor可能是一个数据feeder或别的tensorflow模型的输出
```python
# 初始化我们的Keras模型
model = Sequential()
first_layer = Dense(32, activation='relu', input_dim=784)
model.add(Dense(10, activation='softmax'))
```

你只需要在实例化该模型后，使用set_input来修改首层的输入，然后将剩下模型搭建于其上：
```python
# 初始化我们的Keras模型
model = Sequential()
first_layer = Dense(32, activation='relu', input_dim=784)
first_layer.set_input(my_input_tensor) # 现在首层的输入为 my_input_tensor

# 重新构建模型
model.add(first_layer)
model.add(Dense(10, activation='softmax'))
```

在这个阶段，你可以调用model.load_weights(weights_file)来加载预训练的权重

然后，你或许会收集该模型的输出张量：
```python
output_tensor = model.output
```

**对TensorFlow张量中调用Keras模型**

Keras模型与Keras层的行为一致，因此可以被调用于TensorFlow张量上：

```python
from keras.models import Sequential

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

# 直接将TensorFlow的Tensor传入Keras所构建的网络中
x = tf.placeholder(tf.float32, shape=(None, 784))
y = model(x) 
```

**注意，调用模型时你同时使用了模型的结构与权重，当你在一个tensor上调用模型时，你就在该tensor上创造了一些操作，这些操作重用了已经在模型中出现的TensorFlow变量的对象**

### 多GPU和分布式训练

**将Keras模型分散在多个GPU中训练**

TensorFlow的设备作用域完全与Keras的层和模型兼容，因此你可以使用它们来将一个图的特定部分放在不同的GPU中训练，这里是一个简单的例子：
```python
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)  # all ops in the LSTM layer will live on GPU:0

with tf.device('/gpu:1'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)  # all ops in the LSTM layer will live on GPU:1
```

注意，由LSTM层创建的变量将不会生存在GPU上，不管TensorFlow变量在哪里创建，它们总是生存在CPU上，TensorFlow将隐含的处理设备之间的转换

如果你想在多个GPU上训练同一个模型的多个副本，并在多个副本中进行权重共享，首先你应该在一个设备作用域下实例化你的模型或层，然后在不同GPU设备的作用域下多次调用该模型实例，如：
```python
with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, 784))

    # shared model living on CPU:0
    # it won't actually be run during training; it acts as an op template
    # and as a repository for shared variables
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=784))
    model.add(Dense(10, activation='softmax'))

# replica 0
with tf.device('/gpu:0'):
    output_0 = model(x)  # all ops in the replica will live on GPU:0

# replica 1
with tf.device('/gpu:1'):
    output_1 = model(x)  # all ops in the replica will live on GPU:1

# merge outputs on CPU
with tf.device('/cpu:0'):
    preds = 0.5 * (output_0 + output_1)

# we only run the `preds` tensor, so that only the two
# replicas on GPU get run (plus the merge op on CPU)
output_value = sess.run([preds], feed_dict={x: data})
```
