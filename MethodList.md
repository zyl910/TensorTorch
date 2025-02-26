# Method list

## TTorch

Methods:

- Arange: This method returns a tensor containing values from a given interval [start, end) with a specified step size. When the step size is not an integer, floating-point rounding errors may occur, so it is recommended to subtract a small epsilon from the end value for consistency (此方法返回一个张量，其中包含具有指定步长的给定区间 [start，end) 的值。当步长不是整数时，可能会出现浮点舍入错误，因此建议从结束值中减去一个较小的 epsilon 以保持一致性). Like `torch.arange`.
- CreateAndFill: Create and fill with the given value (创建并使用指定值填充).
- FromNDArray: Create tensor by N-dimensional array (根据N维数组创建张量). Like `torch.tensor`.
- Ones: This method creates a tensor of a specified shape, where each element is initialized to the scalar value 1 (此方法用于创建指定形状的张量，其中每个元素都初始化为标量值1). Like `torch.ones`.
- SumTorch: This function is used to compute the sum of all elements in the input tensor. Support dim parameter (此函数用于对输入张量中所有元素计算求和. 支持 dim 参数). Like `torch.sum`.
- ToString: ReadOnlySpan to String.
- Zeros: This method returns a tensor filled with zeros that has a specified shape (此方法返回一个填充有具有指定形状的零的张量). Like `torch.zeros`.
- ZerosLike: Returns a tensor filled with the scalar value 0, with the same size as input (返回一个填充了标量值0的张量，其大小与 input 相同). Like `torch.zeros_like`.

Extension Methods:

- Clone: Returns a copy of source data (返回源数据的拷贝).
- FillRange: Fills the contents of this ranges with the given value (用给定值填充此范围的内容).
- SliceTorch: Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
- To1DArray: Create 1-dimensional array by tensor. It can also convert N-dimensional tensor into 1-dimensional arrays (根据张量创建1维数组. 它还能将多维张量转为1维数组).
- To2DArray: Create 2-dimensional array by tensor. The rank of a tensor must be equal to 2 (根据张量创建2维数组. 张量的秩必须为 2).
- To3DArray: Create 3-dimensional array by tensor. The rank of a tensor must be equal to 3 (根据张量创建3维数组. 张量的秩必须为 3).
- To4DArray: Create 4-dimensional array by tensor. The rank of a tensor must be equal to 4 (根据张量创建4维数组. 张量的秩必须为 4).
- ToNDArray: Create N-dimensional array by tensor  (根据张量创建N维数组).

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

Member of  tensor object (张量对象的成员).

| PyTorch      | .NET                     | Remark                                   |
| ------------ | ------------------------ | ---------------------------------------- |
| `__str__(x)` | x.ToString               | To string (转为字符串).                       |
| x[indices]   | x[indices]               | Get or set element (读写元素).               |
| x[ranges]    | x.SliceTorch             | Slice (切片). The Slice method needs to have the same dimensions. And SliceTorch method is similar to PyTorch, allowe parameters to have fewer dimensions than they actually are (Slice 方法需要维度相同. 而 SliceTorch 方法类似 PyTorch，允许参数的维度比实际的少). |
| x[ranges]=C  | `X.FillRange(C, ranges)` | Fill slice (填充切片). Or `X.AsTensorSpan()[ranges].FillRange(C)`. |
| x.clone      | x.Clone                  | Clone(克隆).                               |
| x.item       | x.GetPinnableReference   | To scalar (转为标量).                        |
| x.numel      | x.FlattenedLength        | Flattened length (平整后的总长度).              |
| x.reshape    | x.Reshape                | Reshape (变形).                            |
| x.shape      | x.Lengths                | Lengths (各维的长度).                         |
| x.stride     | X.Strides                | Strides (各维的跨距).                         |
| x.sum        | x.SumTorch               | Sum by dimensions (根据维度的求和)              |
| x.T          | Tensor.Transpose         | Transpose (矩阵转置).                        |

Operator (运算符).

| PyTorch | .NET            | Remark       |
| ------- | --------------- | ------------ |
| +       | Tensor.Add      | Add (加)      |
| -       | Tensor.Subtract | Subtract (减) |
| *       | Tensor.Multiply | Multiply (乘) |
| /       | Tensor.Divide   | Divide (除)   |
| **      | Tensor.Pow      | Pow (幂)      |
| ==      | Tensor.Equals   | Equals (相等)  |