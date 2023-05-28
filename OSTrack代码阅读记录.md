

2023.3.16

**配置环境**

。。。。。。。。。

```python
# 单卡训练
CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode single

CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_got10k_ep100 --save_dir ./output --mode single

# 多卡训练
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0


CUDA_VISIBLE_DEVICES=1 python run_training.py --script vipt --config deep_rgbe --save_dir ./output --use_lmdb 0 --script_prv None --config_prv baseline --use_wandb 0 --distill 0 --script_teacher None --config_teacher None

```
```python
CUDA_VISIBLE_DEVICES=2,3 python tracking/test.py --tracker_name ostrack --tracker_param vitb_256_mae_ce_32x4_got10k_ep100 --dataset_name got10k_test --threads 4 --num_gpus 2

CUDA_VISIBLE_DEVICES=0 python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep100

"""
CUDA_VISIBLE_DEVICES=0 python tracking/test.py  ostrack vitb_256_mae_ce_32x4_got10k_ep100 --dataset_name got10k_test --threads 2 --num_gpus 1
"""
```
- GOT10K-test
```python
​```python
# 测试
CUDA_VISIBLE_DEVICES=1 python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 2 --num_gpus 2
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 2 --num_gpus 2
```

**代码报错**

**1、**lib/train/dataset/got10k.py --67行

![1678945496410](C:\Users\52675\AppData\Roaming\Typora\typora-user-images\1678945496410.png)

错误:列表索引超出范围

自己傻逼了。。。

将got10k的序列打乱训练的，自己只给了500个序列，总共有9335序列

解决方法：将9335序列补齐



**2、**加载模型的时候卡在那

![1678952709876](C:\Users\52675\AppData\Roaming\Typora\typora-user-images\1678952709876.png)

继续Debug找原因

**ViT整体结构**

![1679032712790](C:\Users\52675\AppData\Roaming\Typora\typora-user-images\1679032712790.png)



2022.3.17

**1、训练的时候报错**

![1679038952353](C:\Users\52675\AppData\Roaming\Typora\typora-user-images\1679038952353.png)

解决办法：

第一个问题去yaml设置文件中 将 num_worker 设置为0

```
NUM_WORKER: 0
```

![1679039031047](C:\Users\52675\AppData\Roaming\Typora\typora-user-images\1679039031047.png)



**2、net.cuda()卡死**

解决办法：

环境问题

换CUDA （服务器网络不好，实在磨人）



**3、ValueError: The number of weights does not match the population**

![1679059299323](C:\Users\52675\AppData\Roaming\Typora\typora-user-images\1679059299323.png)

替换 （因为测试运行时只用了一个数据集 GOT10k）

```c
dataset = self.datasets[0]
```





2023.3.21

```python
os.path.dirname #功能：去掉文件名，返回目录
os.path.join #连接两个或更多的路径名组件
isinstance(img_size, tuple) #如果对象img_size的类型与参数二的类型（tuple）相同则返回 True，否则返回 False。。
```



2023.3.22

### tensor.view()

```python
patch_pos_embed.shape #-> torch.Size([1, 768, 196])
B, E, Q = patch_pos_embed.shape
patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)  # -> torch.Size([1, 768, 14, 14])
#在pytorch中view函数的作用为重构张量的维度reshape
```



**python:flatten详解**

flatten()是对多维数据的降维函数。
flatten()，默认缺省参数为0，也就是说flatten()和flatten(0)效果一样。
python里的flatten(dim)表示，从第dim个维度开始展开，将后面的维度转化为一维。也就是说，只保留dim之前的维度，其他维度的数据全都挤在dim这一维。

```python
import torch
a = torch.rand(2,3,4)
b = a.flatten()
print(b)
​```
tensor([0.1439, 0.8630, 0.9784, 0.7685, 0.2959, 0.7467, 0.7823, 0.3222, 0.2259,
        0.5248, 0.1076, 0.9935, 0.2795, 0.7154, 0.9743, 0.6546, 0.7482, 0.2878,
        0.8626, 0.3516, 0.7923, 0.8408, 0.3001, 0.3754])
​```
b = a.flatten(1)
print(b)
tensor([[0.1439, 0.8630, 0.9784, 0.7685, 0.2959, 0.7467, 0.7823, 0.3222, 0.2259,
         0.5248, 0.1076, 0.9935],
        [0.2795, 0.7154, 0.9743, 0.6546, 0.7482, 0.2878, 0.8626, 0.3516, 0.7923,
         0.8408, 0.3001, 0.3754]])

b = a.flatten(2)
print(b)
tensor([[[0.1439, 0.8630, 0.9784, 0.7685],
         [0.2959, 0.7467, 0.7823, 0.3222],
         [0.2259, 0.5248, 0.1076, 0.9935]],

        [[0.2795, 0.7154, 0.9743, 0.6546],
         [0.7482, 0.2878, 0.8626, 0.3516],
         [0.7923, 0.8408, 0.3001, 0.3754]]])
```



**[pytorch torch.nn.functional实现插值和上采样]**

### interpolate

```
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
```

根据给定的size或scale_factor参数来对输入进行下/上采样

使用的插值算法取决于参数mode的设置

支持目前的temporal(1D, 如向量数据), spatial(2D, 如jpg、png等图像数据)和volumetric(3D, 如点云数据)类型的采样数据作为输入，输入数据的格式为minibatch x channels x [optional depth] x [optional height] x width，具体为：

- 对于一个temporal输入，期待着3D张量的输入，即minibatch x channels x width
- 对于一个空间spatial输入，期待着4D张量的输入，即minibatch x channels x height x width
- 对于体积volumetric输入，则期待着5D张量的输入，即minibatch x channels x depth x height x width

可用于重置大小的mode有：最近邻、线性(3D-only),、双线性, 双三次(bicubic,4D-only)和三线性(trilinear,5D-only)插值算法和area算法

**参数：**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入张量
- **size** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* *Tuple**[*[*int*](https://docs.python.org/3/library/functions.html#int)*] or* *Tuple**[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*int*](https://docs.python.org/3/library/functions.html#int)*] or* *Tuple**[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*int*](https://docs.python.org/3/library/functions.html#int)*,* [*int*](https://docs.python.org/3/library/functions.html#int)*]*) –输出大小.
- **scale_factor** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* *Tuple**[*[*float*](https://docs.python.org/3/library/functions.html#float)*]*) – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型
- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – `可使用的上采样算法，有'nearest', 'linear', 'bilinear', 'bicubic' , 'trilinear'和'area'. 默认使用'nearest'`
- **align_corners** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) –
 ```
几何上，我们认为输入和输出的像素是正方形，而不是点。如果设置为True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。如果设置为False，则输入和输出张量由它们的角像素的角点对齐，插值使用边界外值的边值填充;当scale_factor保持不变时，使该操作独立于输入大小。仅当使用的算法为'linear', 'bilinear', 'bilinear'or 'trilinear'时可以使用。默认设置为False
 ```

```python
nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
```



2023.3.23

### numpy数组切片操作

通过冒号分隔切片参数 start:stop:step 来进行切片操作：
形如：

```python
b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 21
#冒号 : 的解释：如果只放置一个参数，如 [2]，将返回与该索引相对应的单个元素。如果为 [2:]，表示从该索引开始以后的所有项都将被提取。如果使用了两个参数，如 [2:7]，那么则提取两个索引(不包括停止索引)之间的项。
```

索引序号： 0 1 2 3……

先定义一个二维数组

```python
import numpy as np
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
>>[[1 2 3]
>>[3 4 5]
>>[4 5 6]]
```

```python
>>> a[1:]
array([[3, 4, 5],
      [4, 5, 6]])

```

出现了冒号，意思是从**索引1**（第二个索引）开始到最后，这里指的也是行。冒号后面没有数就是指最大的。冒号前面没有数，就是指最小数0。如下

```python
>>> a[:2]
array([[1, 2, 3],
      [3, 4, 5]])
```

这里的意思就是，从**索引0**开始，到**第2个索引**，也就是索引1。

 ### numpy数组进阶up

```python
>>> a[:,1]
array([2, 4, 5])
```

逗号前面是行，行都没有指定数，也就是说对行每要求，只考虑列了。

```python
>>> a[:,1:3]  #第2列到第3列
array([[2, 3],
      [4, 5],
      [5, 6]])
# 从第几个列索引开始到第几列
>>> a[:,0:2] #第1列到第2列
array([[1, 2],
      [3, 4],
      [4, 5]])
>>> a[:,]  #对列也不操作，跟下面等价
array([[1, 2, 3],
      [3, 4, 5],
      [4, 5, 6]])
>>> a[:,:]
array([[1, 2, 3],
      [3, 4, 5],
      [4, 5, 6]])
```


### tensor.repeat()

```python
a = torch.randn(1,3)
print(a)
#tensor([[-1.1554, -1.2661, -1.0998]])
b = a.repeat(3,1)
print(b)
#tensor([[-1.1554, -1.2661, -1.0998],
#        [-1.1554, -1.2661, -1.0998],
#        [-1.1554, -1.2661, -1.0998]])
```



2023.3.25

### **round()** 方法返回浮点数x的四舍五入值。

### tensor.unsqueeze()

```python
a.unsqueeze(N)
#a.unsqueeze(N) 就是在a中指定位置N加上一个维数为1的维度
```



### torch.clamp()

```python
torch.clamp(input, min, max, out=None) → Tensor
```

限幅。将input的值限制在[min, max]之间，并返回结果。



### torch.max(input, dim) 函数

**输入**

```python
# input是softmax函数输出的一个`tensor` 
# dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
```


**输出**

```python
# 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
```



###  torch.arange() 、torch.range() 、torch.linspace()

根据步长创建一维tensor

```python
# torch.arange()为左闭右开，即[start, end)
torch.arange(5)
# tensor([ 0,  1,  2,  3,  4])
torch.arange(1, 4)
# tensor([ 1,  2,  3])

# torch.range()为左闭右闭，即[start, end]
torch.range(1, 4)
#tensor([ 1.,  2.,  3.,  4.])

torch.linspace(0, 63, 64)
"""
torch.linspace(start,end, steps=64)
start：开始值
end：结束值
steps：分割的点数，默认是100
dtype：返回值（张量）的数据类型
"""

tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
        28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
        42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
        56., 57., 58., 59., 60., 61., 62., 63.])
#  函数的作用是，返回一个一维的tensor（张量），这个张量包含了从start到end，分成steps个线段得到的向量。常用的几个变量
```



###  torch.clamp_min()

设置一个下限min，tensor中有元素小于这个值, 就把对应的值赋为min

```python
tensor([ 130,  144,  183,  609,  818,  832,  891, 1386, 1494, 1729, 1739, 1785,
        2671, 2802, 3444, 4711, 4754, 4763, 5728, 5842])
torch.clamp_min_(1000)
tensor([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1386, 1494, 1729, 1739, 1785,
        2671, 2802, 3444, 4711, 4754, 4763, 5728, 5842])
```



### numpy中np.finfo用法

```python
"""

np.finfo使用方法

eps是一个很小的非负数

除法的分母不能为0的,不然会直接跳出显示错误。

使用eps将可能出现的零用eps来替换，这样不会报错。

"""

eps = np.finfo(x.dtype).eps # eps = 2.220446049250313e-16 type = <class 'numpy.float64'>

```



2023.4.3

### divmod() 函数

```python
divmod(a, b)
>>>divmod(7, 2)
(3, 1)
>>> divmod(8, 2)
(4, 0)
#python divmod() 函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组(a // b, a % b)
```



