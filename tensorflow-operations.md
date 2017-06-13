

```python
import tensorflow as tf
```

# TensorflowBoard

### Save the graph to local file


```python
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
```

    5


### use tensorboard to load the graph 
$ tensorboard --logdir="./graph"  
Then go to browser and open http://localhost:6006

# Constant Types


```python
# constant of 0d tensor
a = tf.constant(1, name="a")
print(a)
```

    Tensor("a_1:0", shape=(), dtype=int32)



```python
# constant of 1d tensor (vector)
b = tf.constant([1, 2], name="b")
b
```




    <tf.Tensor 'b:0' shape=(2,) dtype=int32>




```python

```


```python
c = tf.zeros([2, 3], tf.int32)
print(c)
```

    Tensor("zeros:0", shape=(2, 3), dtype=int32)



```python
d = tf.zeros_like([[1, 2], [3, 4]])
print(d)
```

    Tensor("zeros_like:0", shape=(2, 2), dtype=int32)



```python

```


```python
with tf.Session() as sess:
    print(a.eval())
    print(b.eval())    
    print(c.eval()) 
```

    1
    [1 2]
    [[0 0 0]
     [0 0 0]]


# Variables


```python
a = tf.Variable(2, name="scalar")
b = tf.Variable([0, 1], name="vecor")
c = tf.Variable(tf.zeros([784, 10]))
```

### Have to initialize the variables before using them


```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

### To get the value of a variable, we need to evaluate it using eval()


```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    print(a)
    print(a.eval()) # Need to evaluate in a session
```

    Tensor("scalar/read:0", shape=(), dtype=int32)
    2


### Each session has its own current value for a variable defined in a graph


```python
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2)))  # >> 8

sess1.close()
sess2.close()
```

    20
    8


# InteractiveSession


```python
a = tf.constant(1)
a.eval() # WRONG!!!
```


```python
b = tf.constant(2)
with tf.Session() as sess:
    print(b.eval()) # correct, evaluate in a session
```

    2



```python
sess = tf.Session()
c = tf.constant(3)
print(c.eval(session=sess)) # correct. ecplicit declare a session

sess.close()
```

    3



```python
sess = tf.InteractiveSession()
d = tf.constant(4)
print(d.eval()) # correct, because we have define a default session

sess.close()
```

    4


# placeholder and feed_dict


```python
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([2, 3, 4], tf.float32)
print(a)
print(b)
```

    Tensor("Placeholder:0", shape=(3,), dtype=float32)
    Tensor("Const_5:0", shape=(3,), dtype=float32)



```python
c = a + b
```


```python
with tf.Session() as sess:
    print(sess.run(c))  # WRONG!!! Need to feed value to the variable first.
```


```python
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))
```

    [ 3.  5.  7.]



```python

```
