# TensorTorch

Add PyTorch style functions to Tensor type (为张量类型补上 PyTorch 风格的函数).

The tensor type is from [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors) (张量类型出自 [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors)).

Commonly Used Types (常用类型):

- Tensor

Python samples from ["Dive into Deep Learning"](https://c.d2l.ai/gtc2020/) (Python 样例出自 [《动手学深度学习》](https://courses.d2l.ai/zh-v2)).

## Mapping to PyTorch Method (PyTorch 方法的对照)

Member of torch (torch 的成员).

| PyTorch    | .NET                                     | Remark                                   |
| ---------- | ---------------------------------------- | ---------------------------------------- |
| arange     | TTorch.Arange                            | Create by range (根据范围创建).                |
| cat        | Tensor.Concatenate, Tensor.ConcatenateOnDimension | Concatenate (连接)                         |
| exp        | Tensor.Exp                               | Natural exponential function (自然指数函数)    |
| ones       | TTorch.Ones                              | Create and fill 1 (创建并填充1).              |
| randn      | Tensor.CreateAndFillGaussianNormalDistribution | Creates and initializes it with random data in a gaussian normal distribution (创建并使用高斯正态分布的随机数据初始化). |
| sum        | Tensor.Sum                               | Sum (求和)                                 |
| tensor     | TTorch.FromNDArray                       | Create tensor by N-dimensional array (根据N维数组创建张量) |
| zeros      | TTorch.Zeros                             | Create and fill 0 (创建并填充0).              |
| zeros_like | TTorch.ZerosLike                         | Returns a tensor filled with the scalar value 0, with the same size as input (返回一个填充了标量值0的张量，其大小与 input 相同) |

- Full method list: [MethodList](MethodList.md)
