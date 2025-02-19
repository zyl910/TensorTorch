# Method list

## TTorch

Methods:

- Arange: This method returns a tensor containing values from a given interval [start, end) with a specified step size. When the step size is not an integer, floating-point rounding errors may occur, so it is recommended to subtract a small epsilon from the end value for consistency (此方法返回一个张量，其中包含具有指定步长的给定区间 [start，end) 的值。当步长不是整数时，可能会出现浮点舍入错误，因此建议从结束值中减去一个较小的 epsilon 以保持一致性). Like `torch.arange`.
- CreateAndFill: Create and fill with the given value (创建并使用指定值填充).
- FromNDArray: Create tensor based on ND array (根据N维数组创建张量). Like `torch.tensor`.
- Ones: This method creates a tensor of a specified shape, where each element is initialized to the scalar value 1 (此方法用于创建指定形状的张量，其中每个元素都初始化为标量值1). Like `torch.ones`.
- ToString: ReadOnlySpan to String.
- Zeros: This method returns a tensor filled with zeros that has a specified shape (此方法返回一个填充有具有指定形状的零的张量). Like `torch.zeros`.

Extension Methods:

- FillRange: Fills the contents of this ranges with the given value (用给定值填充此范围的内容).
- SliceTorch: Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).

## Map to PyTorch Method (PyTorch 方法的对照)

Member of torch (torch 的成员).

| PyTorch | .NET                                     | Remark |
| ------- | ---------------------------------------- | ------ |
| arange  | TTorch.Arange                            |        |
| cat     | Tensor.Concatenate, Tensor.ConcatenateOnDimension |        |
| exp     | Tensor.Exp                               |        |
| ones    | TTorch.Ones                              |        |
| randn   | Tensor.CreateAndFillGaussianNormalDistribution |        |
| sum     | Tensor.Sum                               |        |
| tensor  | TTorch.FromNDArray                       |        |
| zeros   | TTorch.Zeros                             |        |
|         |                                          |        |

Member of  tensor object (张量对象的成员).

| PyTorch      | .NET                     | Remark                                   |
| ------------ | ------------------------ | ---------------------------------------- |
| `__str__(x)` | x.ToString               | To string.                               |
| x[indices]   | x[indices]               | Get or set item.                         |
| x[ranges]    | x.SliceTorch             | The Slice method needs to have the same dimensions. And SliceTorch method is similar to PyTorch, allowe parameters to have fewer dimensions than they actually are (Slice 方法需要维度相同. 而 SliceTorch 方法类似 PyTorch，允许参数的维度比实际的少). |
| x[ranges]=C  | `X.FillRange(C, ranges)` | Or `X.AsTensorSpan()[ranges].FillRange(C)`. |
| x.item       | x.GetPinnableReference   |                                          |
| x.numel      | x.FlattenedLength        |                                          |
| x.reshape    | x.Reshape                |                                          |
| x.shape      | x.Lengths                |                                          |

Operator (运算符).

| PyTorch | .NET            | Remark |
| ------- | --------------- | ------ |
| +       | Tensor.Add      |        |
| -       | Tensor.Subtract |        |
| *       | Tensor.Multiply |        |
| /       | Tensor.Divide   |        |
| **      | Tensor.Pow      |        |
| ==      | Tensor.Equals   |        |