---
layout: post
title:  TensorFlow变量作用域机制
categories: TensorFlow
tags:   TensorFlow 机器学习 
author: 陈权
---

* content
{:toc}





### TensorFlow变量作用域机制

#### **1、tf.variable_scope()**
`tf.variable_scope()` 变量作用域机制在 TensorFlow 中主要由两部分组成：
- 当`tf.get_variable_scope().reuse == False` 时， variable_scope 作用域只能用来创建新变量
- 当 `tf.get_variable_scope().reuse == True` 时，variable_scope 作用域可以用来创建新变量和共享变量

**a、tf.get_variable_scope().reuse == False**

```python
with tf.variable_scope("foo"): # 此时reuse默认为False，应此不能使用同一个变量名在此作用域下申请变量
	v = tf.get_variable("v", [1])
	v2 = tf.get_variable("v", [1])
```

上述程序`v2 = tf.get_variable("v", [1])`会报错，因为使用了同一个变量名申请变量。

**b、tf.get_variable_scope().reuse == True**



```python
with tf.variable_scope("foo") as scope:
	v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    #也可以写成：
    #scope.reuse_variables()
	v1 = tf.get_variable("v", [1])
	assert v1 == v
```



在tensorflow中，为了 **`节约变量存储空间`和`层级参数共享`** ，我们常常需要通过共享 **变量作用域(variable_scope)** 来实现 **共享变量** 。

大家比较常用也比较笨的一种方法是，在重复使用（即 **非第一次使用**）时，设置**` reuse=True `**来 **再次调用** 该共享变量作用域（variable_scope）。但是这种方法太繁琐了。

下面我们使用一种简便的方法：

**使用`tf.AUTO_REUSE`**

```python
w=dict()
def f(sess,i):
    with tf.variable_scope(name_or_scope='foo', reuse=tf.AUTO_REUSE):    ### 改动部分 ###
        tf.set_random_seed(2018)
        w[str(i)+"w"] = tf.get_variable("v", [1],initializer=tf.ones_initializer())
        print(str(i)+"w",sess.run(w[str(i)+"w"]),w[str(i)+"w"].name)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print("===")
        f(sess,i)
```

官方文档案例:

```python
def foo():
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        v = tf.get_variable("v", [1])
    return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2
```

**获取变量作用域**

可以直接通过 `tf.variable_scope()`来获取变量作用域：    

```python
with tf.variable_scope("foo") as foo_scope:
	v = tf.get_variable("v", [1])
with tf.variable_scope(foo_scope)
	w = tf.get_variable("w", [1])
```



如果在开启的一个`变量作用域`里使用之前预先定义的一个作用域，则会跳过当前变量的作用域，保持预先存在的作用域不变。    

```python
with tf.variable_scope("foo") as foo_scope:
	assert foo_scope.name == "foo"
with tf.variable_scope("bar")
	with tf.variable_scope("baz") as other_scope:
		assert other_scope.name == "bar/baz"
		with tf.variable_scope(foo_scope) as foo_scope2:
			assert foo_scope2.name == "foo" # 保持不变
```

**变量作用域的初始化**

变量作用域可以默认携带一个`初始化器`，在这个作用域中的子作用域或变量都可以`继承`或者`重写`父作用域初始化器中的值。    

```python
with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):
	v = tf.get_variable("v", [1])
	assert v.eval() == 0.4 # 被作用域初始化
	w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3)):
	assert w.eval() == 0.3 # 重写初始化器的值
	with tf.variable_scope("bar"):
		v = tf.get_variable("v", [1])
		assert v.eval() == 0.4 # 继承默认的初始化器
	with tf.variable_scope("baz", initializer=tf.constant_initializer(0.2)):
		v = tf.get_variable("v", [1])
		assert v.eval() == 0.2 # 重写父作用域的初始化器的值
```

上面讲的是 variable_name，那对于 op_name 呢？在 tf.variable_scope() 作用域下的操作，也会 被加上前缀：

```python
with tf.variable_scope("foo"):
	x = 1.0 + tf.get_variable("v", [1])
    # x = tf.add(1.0,tf.get_variable("v", [1]))  # 等价
    assert x.op.name == "foo/add"
```

> **`variable_scope `主要用在循环神经网络（RNN）和卷积神经网络（CNN）人脸识别的操作中，其中需要大量的共享变量。** 

#### **2、tf.variable_scope()**

TensorFlow 中常常会有数以千计的节点，在可视化的过程中很难一下子展示出来，因此用 `tf.name_scope()` 为变量划分范围，在可视化中，这表示在计算图中的一个层级。 `tf.name_scope() `会影响 op_name，不会影响用 get_variable()创建的变量，而会影响通过 `tf.Variable()`创建的变量。    



```python
with tf.variable_scope("foo"):
	with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        b = tf.Variable(tf.zeros([1]), name='b')
        x = 1.0 + v
assert v.name == "foo/v:0"
assert b.name == "foo/bar/b:0"
assert x.op.name == "foo/bar/add"
```

可以看出，` tf.name_scope()`返回的是一个字符串，如上述的"bar"。 name_scope 对用`tf.get_variable()`创建的变量的名字不会有任何影响，而` tf.Variable()`创建的操作会被加上前缀，并且会给操作加上名字前缀。