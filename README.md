边学边总结，更新页面见 [Github](https://hfut-zyw.github.io/Pytorch-Tutorial/)  
访问我的主页地址 [Homepage](https://hfut-zyw.github.io/)

![](Torch.png)

## Pytorch的核心模块说明

- Pytorch两大主要功能：计算图构建与反向传播，优化 
- 计算图构建与反向传播实现方案：定义各种operator节点类（Variable，Function，Module），每个operator实现forward和backward方法 ，data区使用Tensor存储
- 优化器的实现方案：基于object提供各种优化器类 
- [Pytorch源码](https://github.com/pytorch/pytorch/tree/master/torch)

### autograd模块 
- [doc](https://pytorch.org/docs/stable/autograd.html)
- 定义Variable节点，实现基本的加减乘除算子操作和反向传播（最新的torch已经不区分Variable和Tensor了，文档里对tensor类也加入了前向反向传播） 
- 定义Function类，作为复杂的算子的基类；需要用户自己写更为复杂的算子，autograd中没有提供写好的算子包
- 上述实现操作和反向传播方法的时候需要调用functional中的API，而functional中的API又是调用C++的底层实现

### nn模块 
- [doc](https://pytorch.org/docs/stable/nn.html)
- 从零开始定义Module类，作为各种Layer和Loss的基类 
- 基于Module扩充为各种Layer和Loss，实现神经单元的前向传播和反向传播算法
- 上述实现操作和反向传播方法的时候需要调用functional中的API

### optim模块 
- [doc](https://pytorch.org/docs/stable/optim.html)
- 基于object定义Optimizer类，作为所有优化器的基类 
- 基于Optimizer实现各种优化器，SGD,ADAD,ADMM等  

### 计算图与优化器的交互 
- 对于基础模块autograd，只能将Variable节点传递给优化器 
- 对于nn模块，Module提供了parameter方法收集Variable参数，并传递给优化器
- parameter方法使用了Parameter容器类来存储参数，Parameter类是Tensor的子类。也是一个Variable节点，但是不参与计算图构建，不需要实现forward和backward方法

### Homework 1
- 基于Function写出exp（）算子 
- 使用autograd提供的Variable，Function构建一个简单的神经网络 
- 对某个叶子节点参数进行优化 
### Homeworak 2
- 基于Module写出一个简单的神经元 
- 使用nn模块提供的Module，各种Layer构建一个简单的神经网络
- 对神经网络的所有参数进行优化 
### Homeworak 3
- 在Homework2的基础上基于Module模块看懂Transformer论文源码 
[Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer)

# <a id="Top">目录</a>


* <a href="#Tensor">*矩阵(Tensor)*</a>
   * <a href="#Tensor0">1.矩阵类型</a>
   * <a href="#Tensor1">2.矩阵创建</a>
   * <a href="#Tensor2">3.矩阵切片</a>
   * <a href="#Tensor3">4.简单运算</a>
   * <a href="#Tensor4">5.矩阵变形</a>
   * <a href="#Tensor5">6.求和均值方差</a>
   * <a href="#Autograd">7.自动求导</a>
   * <a href="#GPU">8.GPU的使用</a>
* <a href="#Dataset/DataLoader">*数据集(Dataset/DataLoader)*</a>
* <a href="#Loss-function">*损失函数(Loss-function)*</a>
* <a href="#Optimizer">*优化器(Optimizer)*</a>
* <a href="#Model">*模型搭建(Model)*</a>
* <a href="#Train/Test">*训练/测试(Train/Test)*</a>
* <a href="#Visualization">*可视化(Visualization)*</a>

---
---
---

## <a id="Tensor">*矩阵(Tensor)*</a>

---


```python
import numpy as np
import torch
```

### <a id="Tensor0">1.矩阵类型</a> 
* Pytorch中定义了8种CPU张量类型和对应的GPU张量类型  
  
* torch.Tensor()、torch.rand()、torch.randn() 均默认生成 torch.FloatTensor型  
  
* 相同数据类型的tensor才能做运算


|    数据类型    |            dytpe            |        CPU         |           GPU           |
| :------------: | :-------------------------: | :----------------: | :---------------------: |
|   16位浮点型   |             //              |         //         |           //            |
|   32位浮点型   | torch.float32或torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
|   64位浮点型   | torch.float64或torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
|                |                             |                    |                         |
| 8位无符号整型  |         torch.uint8         |         //         |           //            |
| 8位有符号整型  |         torch.int8          |         //         |           //            |
| 16位有符号整型 |         torch.int16         |         //         |           //            |
| 32位有符号整型 |         torch.int32         |         //         |           //            |
| 64位有符号整型 |         torch.int64         |         //         |           //            |


* 全局矩阵类型设置    
  torch.set_default_tensor_type(torch.FloatTensor)  
  
* 数据类型转换  
在Tensor后加 .long(), .int(), .float(), .double()等即可，也可以用.to()函数进行转换，也可以在创建是填写dtype参数指定类型  

* 与numpy数据类型转换  
Tensor---->Numpy 使用 data.numpy()，data为Tensor变量  
Numpy ----> Tensor 使用 torch.from_numpy(data)，data为numpy变量  

* 与Python数据类型转换  
Tensor ----> 单个Python数据，使用data.item()，data为Tensor变量且只能为包含单个数据  
Tensor ----> Python list，使用data.tolist()，data为Tensor变量，返回shape相同的可嵌套的list  


---

### <a id="Tensor1">2.矩阵创建</a>

<font color=Green>从list,numpy创建</font>

- torch.tensor(),&emsp;torch.from_numpy()


```python
x=torch.tensor([[1,2,3,4],[2,3,4,5]],dtype=torch.int32)
y=torch.tensor([[1,2,3,4],[2,3,4,5]],dtype=torch.float32)
z=torch.from_numpy(np.array([[1,2,3,4],[2,3,4,5]]))
x,y,z
```




    (tensor([[1, 2, 3, 4],
             [2, 3, 4, 5]], dtype=torch.int32),
     tensor([[1., 2., 3., 4.],
             [2., 3., 4., 5.]]),
     tensor([[1, 2, 3, 4],
             [2, 3, 4, 5]], dtype=torch.int32))



<font color=Green>从函数创建</font>

 - <font color=MediumPurple >torch.empty(尺寸)&emsp;torch.full(尺寸,值)</font>   
   
 - <font color=MediumPurple >torch.zeros(尺寸)&emsp;torch.ones(尺寸)&emsp;torch.eye(维数)   </font> 
   
 - <font color=MediumPurple >torch.zeros_like(另一个矩阵)&emsp;torch.ones_like(另一个矩阵)  </font>   

note：尺寸可以是一维的，也可以是多维的，一般用列表框起来


```python
a1=torch.empty(3)
a2=torch.empty(3,2)
b=torch.eye(3)   
c1=torch.zeros(3)
c2=torch.zeros(2,3)
d=torch.ones([2,3])
e=torch.full([2,3],6)
f=torch.zeros_like(a1)
a1,a2,b,c1,c2,d,e,f
```




    (tensor([0., 0., 0.]),
     tensor([[2.1270e-07, 1.0357e-11],
             [1.3296e+22, 5.4885e-05],
             [3.2768e-09, 1.0335e-05]]),
     tensor([[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]),
     tensor([0., 0., 0.]),
     tensor([[0., 0., 0.],
             [0., 0., 0.]]),
     tensor([[1., 1., 1.],
             [1., 1., 1.]]),
     tensor([[6, 6, 6],
             [6, 6, 6]]),
     tensor([0., 0., 0.]))



<font color=Green>区间线性采样</font>

- <font color=MediumPurple >torch.arange(首，尾，可选步长) </font>   
  note：不包括尾巴  
  
- <font color=MediumPurple >torch.linspace(首，尾，数量)  </font>  
note：包括尾巴，步长=(尾-首)/(n-1),因为starts+(n-1)step=end


```python
xx=torch.arange(5,8)
yy=torch.arange(5,8,2)
aa=torch.linspace(5,8,1)
bb=torch.linspace(5,8,10)
xx,yy,aa,bb
```




    (tensor([5, 6, 7]),
     tensor([5, 7]),
     tensor([5.]),
     tensor([5.0000, 5.3333, 5.6667, 6.0000, 6.3333, 6.6667, 7.0000, 7.3333, 7.6667,
             8.0000]))



<font color=Green >随机矩阵生成</font>

- <font color=MediumPurple >torch.rand(尺寸)</font>  
均匀分布$U(0,1)$


```python
torch.rand(3),torch.rand(3,4)
```




    (tensor([0.5664, 0.2675, 0.0412]),
     tensor([[0.7490, 0.7631, 0.4868, 0.3903],
             [0.2017, 0.3459, 0.5577, 0.7504],
             [0.2212, 0.2027, 0.7184, 0.4905]]))



- <font color=MediumPurple > torch.randn(尺寸)，torch.normal(均值，方差，尺寸) </font>  
正态分布$N(0,1),N(u,\sigma^2)$


```python
torch.randn([3,4]),torch.normal(10,3,[3,4])
```




    (tensor([[ 0.0684, -0.2628,  0.7785, -1.0128],
             [-0.8024, -0.7641,  0.3274,  0.7126],
             [ 0.9564,  0.1232,  0.6068,  1.0955]]),
     tensor([[ 8.9840, 13.2170, 13.1209, 13.3525],
             [16.4026, 10.0551, 10.7741,  9.6713],
             [13.0219, 15.5152,  7.3585,  5.5696]]))



---

### <a id="Tensor2">3.矩阵切片</a>

- 逗号的作用：   
    逗号前表示行，逗号后表示列   
    
- 冒号的作用 
    - 一个冒号：start：end  &nbsp;包括start，不包括end
    - 两个冒号：start：end：step  &nbsp;包含步长

<font color=Green>一维矩阵切片</font>

- 同列表


```python
torch.random.seed()
a=torch.rand(10) 
print(f'  a={a}  ,  a[5]={a[5]}  ,  a[0:3]={a[0:3]}  ,  a[:6]={a[:6]}  ,  a[:-1]={a[:-1]}  ')
```

      a=tensor([0.0042, 0.7859, 0.0226, 0.7523, 0.9846, 0.4688, 0.4635, 0.6405, 0.8597,
            0.9072])  ,  a[5]=0.46884816884994507  ,  a[0:3]=tensor([0.0042, 0.7859, 0.0226])  ,  a[:6]=tensor([0.0042, 0.7859, 0.0226, 0.7523, 0.9846, 0.4688])  ,  a[:-1]=tensor([0.0042, 0.7859, 0.0226, 0.7523, 0.9846, 0.4688, 0.4635, 0.6405, 0.8597])  


<font color=Green>二维矩阵切片</font>


```python
a=torch.rand([6,4])
print(f'a={a}')
```

    a=tensor([[0.9955, 0.0234, 0.2595, 0.4704],
            [0.6995, 0.4418, 0.7450, 0.0307],
            [0.2733, 0.0814, 0.7489, 0.3722],
            [0.2989, 0.3799, 0.1584, 0.9450],
            [0.6366, 0.6279, 0.8427, 0.8612],
            [0.6888, 0.4757, 0.4032, 0.6462]])


- 取行操作


```python
print(f'a[0]={a[0]}')
```

    a[0]=tensor([0.9955, 0.0234, 0.2595, 0.4704])



```python
print(f'a[0:1]={a[0:2]}')
```

    a[0:1]=tensor([[0.9955, 0.0234, 0.2595, 0.4704],
            [0.6995, 0.4418, 0.7450, 0.0307]])



```python
print(f'a[0:-1:2]={a[0:-1:2]}')
```

    a[0:-1:2]=tensor([[0.9955, 0.0234, 0.2595, 0.4704],
            [0.2733, 0.0814, 0.7489, 0.3722],
            [0.6366, 0.6279, 0.8427, 0.8612]])


- 取列操作


```python
print(f'a[:,0]={a[:,0]}')
```

    a[:,0]=tensor([0.9955, 0.6995, 0.2733, 0.2989, 0.6366, 0.6888])


- 综合操作


```python
print(f'a[0:2,-1]={a[0:2,-1]}')
```

    a[0:2,-1]=tensor([0.4704, 0.0307])


---

### <a id="Tensor3">4.简单运算</a>

- 按元素做加减乘除  
    - a+b=torch.add(a,b)  
    - a-b=torch.sub(a,b)  
    - a*b=torch.mul(a,b)  
    - a/b=torch.div(a,b)     
- 按元素运算的广播机制  
    - 1）如果两个张量shape的长度不一致，那么需要在较小长度的shape前添加1，直到两个张量的形状长度相等。  
    - 2） 保证两个张量形状相等之后，每个维度上的结果维度就是当前维度上较大的那个。  
          以张量x和y进行广播为例，x的shape为[2, 3, 1，5]，张量y的shape为[3，4，1]。首先张量y的形状长度较小，  
          因此要将该张量形状补齐为[1, 3, 4, 1]，再对两个张量的每一维进行比较。从第一维看，x在一维上的大小为2，y为1，  
          因此，结果张量在第一维的大小为2。以此类推，对每一维进行比较，得到结果张量的形状为[2, 3, 4, 5]。  
    - 3） 当维数不等的时候，必须有一个维数为1，否则出错  
    <br>
- 矩阵乘法  
    - Wx=torch.matmul(W,x)  
- 矩阵乘法的广播机制   
    - 1）如果两个张量均为一维，则获得点积结果。  
    - 2） 如果两个张量都是二维的，则获得矩阵与矩阵的乘积。  
    - 3） 如果张量x是一维，y是二维，则将x的shape转换为[1, D]，与y进行矩阵相乘后再删除前置尺寸。  
    - 4） 如果张量x是二维，y是一维，则获得矩阵与向量的乘积。  
    - 5） 如果两个张量都是N维张量（N > 2），则根据广播规则广播非矩阵维度（除最后两个维度外其余维度）。  
          比如：如果输入x是形状为[j,1,n,m]的张量，另一个y是[k,m,p]的张量，则输出张量的形状为[j,k,n,p]。  


```python
x=torch.tensor([2,3])
y=torch.tensor([[1,2],[3,4]])
torch.matmul(x,y),torch.matmul(y,x)
```




    (tensor([11, 16]), tensor([ 8, 18]))



---

### <a id="Tensor4">5.矩阵变形</a>

<font color=Green >转置与reshap</font>

- torch.reshape(尺寸)  
    将矩阵拉平后，变成想要的尺寸  
- torch.flatten()  
    将矩阵拉平  
- torch.transpose(dim,dim)  
    指定维数进行转置，没指定的看成整体


```python
x=torch.tensor([[[1,2,3],[6,7,8]]])
x,x.shape,x.shape[1]
```




    (tensor([[[1, 2, 3],
              [6, 7, 8]]]),
     torch.Size([1, 2, 3]),
     2)




```python
x.flatten()
```




    tensor([1, 2, 3, 6, 7, 8])




```python
x.reshape(1,3,-1)
```




    tensor([[[1, 2],
             [3, 6],
             [7, 8]]])




```python
x.transpose(1,2)
```




    tensor([[[1, 6],
             [2, 7],
             [3, 8]]])




```python
x.transpose(0,1)
```




    tensor([[[1, 2, 3]],
    
            [[6, 7, 8]]])



<font color=Green >分割与拼接</font>

- torch.chunk(块数，dim)  
    在指定维数将矩阵分成相应的块数，先看一维矩阵分割，再看多维矩阵的分割


```python
torch.tensor([1,2,3,4,5,6,7,8,9]).chunk(3,0)
```




    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))




```python
x=torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[3,7,0]]])
#第0维包含2个元素(矩阵)，第1维包含2行向量，第2维包含3个元素

x.chunk(2,0)#把第0维的两个矩阵分开，获得两个矩阵
```




    (tensor([[[1, 2, 3],
              [4, 5, 6]]]),
     tensor([[[7, 8, 9],
              [3, 7, 0]]]))



- torch.cat((a,b),dim)
    - dim=0,分别将第0维的元素看成整体，分别拼接;获得的维数为指定维数相加
    - dim=1，分别将第1维的元素看成整体，分别拼接
    - dim=2，分别将第2维的元素看成整体，分别拼接


```python
a,b=x.chunk(2,0)
print(a,'\n',b)
torch.cat((a,b),0),torch.cat((a,b),1)
```

    tensor([[[1, 2, 3],
             [4, 5, 6]]]) 
     tensor([[[7, 8, 9],
             [3, 7, 0]]])





    (tensor([[[1, 2, 3],
              [4, 5, 6]],
     
             [[7, 8, 9],
              [3, 7, 0]]]),
     tensor([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [3, 7, 0]]]))



<font color=Green >squeeze与unsqueeze</font>

- torch.squeeze()
    删除所有维数为1的维度  
- torch.squeeze(dim)  
    删除指定维度为1的维度,其实就是去掉指定维数的括号


```python
x=torch.tensor([[[1],[2],[3]],[[4],[5],[6]]])
x.squeeze(2)
```




    tensor([[1, 2, 3],
            [4, 5, 6]])



- torch.unsqueeze(dim)  
    - 在指定的地方加一个维数，比如dim=1，原来的尺寸为[2,3],变形后为[2,1,3]
    - dim=0,   &nbsp;在这里插入括号[[[]]]  
    - dim=1,   &nbsp;[这里[[]]]  
    - dim=2,   &nbsp;[[这里[]]]


```python
y=x.squeeze(2)
y,y.unsqueeze(1)
```




    (tensor([[1, 2, 3],
             [4, 5, 6]]),
     tensor([[[1, 2, 3]],
     
             [[4, 5, 6]]]))



- 关于维度的小结  
    - [[[分别看作第0，1，2维的墙  
    - 如果对第0维进行操作，比如sum，mean，min的操作，那么进入第0维房间，把里面的元素看成整体，进行操作，操作后只改变第0维的维数  
    - 如果是增加维数（墙面），按指定的位置增加墙面即可，如squeeze和unsqueeze

---

### <a id="Tensor5">6.求和均值方差</a>

- torch.sum() &nbsp; torch.mean() &nbsp; torch.var() &nbsp; torch.std()  
    - 对全部元素求和,均值，方差，标准差  
- torch.sum(dim) &nbsp; torch.mean(dim) &nbsp; torch.var(dim) &nbsp; torch.std(dim)
    - 进入指定的维数房间，对里面的元素进行求和，均值，方差，标准差


```python
x=torch.tensor([[1,2,3],[4,5,6]])
x,x.sum(dim=0),x.sum(dim=1)
```




    (tensor([[1, 2, 3],
             [4, 5, 6]]),
     tensor([5, 7, 9]),
     tensor([ 6, 15]))



---

### <a id="Autograd">7.自动求导</a>

[自动微分机制参考:PaddlePaddle](https://aistudio.baidu.com/aistudio/projectdetail/2528424)

<font color=Green >基本概念</font>
- 这里仅讨论标量对参数的求导，梯度与参数矩阵形状一致，表示每个参数动一动，标量的变化  
- 被求导的变量，需要设置requires_grad=True，dtype=torch.float32  
- 求导的梯度，在参数的grad属性中保存


```python
a=torch.ones([2,3],dtype=torch.float32,requires_grad=True)
b=a.sum()
b.backward()
print(a.grad)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]])


<font color=Green >梯度的累加</font>  
- 梯度每次计算都会被累加在grad属性中  
- 使用torch.grad.zero_()方法可以进行梯度归零  


```python
a=torch.ones([2,3],dtype=torch.float32,requires_grad=True)
for i in range(3):
    b=a.sum()
    b.backward()
    print(a.grad)
a.grad.zero_()
print(a.grad)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[2., 2., 2.],
            [2., 2., 2.]])
    tensor([[3., 3., 3.],
            [3., 3., 3.]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])


<font color=Green >计算图的构建与销毁</font>  
- 每次进行前向计算时都会自动构建计算图，调用backward后自动销毁  
- 在调用backward方法的时候设置retain_graph=True，计算图可保留，不用再次前向计算


```python
a=torch.ones([2,3],dtype=torch.float32,requires_grad=True)
b=a.sum()
b.backward(retain_graph=True) #计算图保留，相当于对计算图什么也没做，就求了一次梯度
print(a.grad)
b.backward() #调用后自动销毁计算图
print(a.grad)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[2., 2., 2.],
            [2., 2., 2.]])


<font color=Green >不构建计算图</font>  
- 前向计算时，构建正向计算图的同时，会通过回溯的方式，构建反向算子与反向计算图  
- 可以通过查看grad_fn属性查看有没有构建反向算子
- with torch.no_grad(): 后面的计算，不会构建计算图  



```python
a=torch.tensor(2.0,requires_grad=True)
f=2*a                                  #f参与构建计算图
g=3*b                                  #g参与构建计算图
with torch.no_grad():
    f=3*a                              #f重新定义，不构建计算图
print(f.grad_fn,g.grad_fn)
```

    None <MulBackward0 object at 0x00000215F129EC70>


<font color=Green >将变量变成常量</font>  
- a.detach()返回一个张量，data区就是a的data区，requires_grad=False  
- 可以用out来接收，输入到下一个网络中，并且作为常量传入的，不参与梯度计算  
- 由于是新创建的张量，不影响原来的计算图 ；但是不要用同一个变量名接收这个张量，否则计算图中的那个变量名就没了 
- 固定网络A的参数，更新网络B的参数的方法  
    - 方法一：将网络A的输入detach一下，创建新的标量张量作为网络B的输入，这时候只会构建B网络的计算图  
    - 方法二：用for遍历A网络的参数，将requires_grad属性全部设为False  
- 一个不常用的torch.detach_()操作，原地修改计算图，有点复杂，实际上所有的tensor操作后加个下划线都是原地操作 


```python
a=torch.tensor(2.0,requires_grad=True)
b=a.detach()
a,b
```




    (tensor(2., requires_grad=True), tensor(2.))



---

### <a id="GPU">8.GPU的使用</a>

- 这里仅使用一块gpu
- 查看设备是否可用：torch.cuda.is_available()查看设备
- 指定设备：device = torch.device(“cuda:0”)或device = torch.device( “cpu”)或device = torch.device(“cuda:0”  if torch.cuda.is_available() else “cpu”)
- 使用tensor.to(device)和model.to(tensor)把输入张量和模型参数送进gpu，这样计算图便在gpu中构建了；使用.device可以查看设备属性


```python
print(torch.cuda.is_available())
torch.tensor(2.0).device
```

    False





    device(type='cpu')



---
---
---

## <a id="Dataset/DataLoader">*数据集(Dataset/DataLoader)*</a>

---
---
---

## <a id="Loss-function">*损失函数(Loss-function)*</a>

---
---
---

## <a id="Optimizer">*优化器(Optimizer)*</a>

---
---
---

## <a id="Model">*模型搭建(Model)*</a>

---
---
---

## <a id="Train/Test">*训练/测试(Train/Test)*</a>

---
---
---

## <a id="Visualization">*可视化(Visualization)*</a>

<a href="#Top">To the top</a>
