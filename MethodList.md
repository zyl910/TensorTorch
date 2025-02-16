# Method list

## TTorch

- Arange: This method returns a tensor containing values from a given interval [start, end) with a specified step size. When the step size is not an integer, floating-point rounding errors may occur, so it is recommended to subtract a small epsilon from the end value for consistency (此方法返回一个张量，其中包含具有指定步长的给定区间 [start，end) 的值。当步长不是整数时，可能会出现浮点舍入错误，因此建议从结束值中减去一个较小的 epsilon 以保持一致性).
- CreateAndFill: Create and fill with the given value (创建并使用指定值填充).
- Ones: This method creates a tensor of a specified shape, where each element is initialized to the scalar value 1 (此方法用于创建指定形状的张量，其中每个元素都初始化为标量值1). 
- ToString: ReadOnlySpan to String.
- Zeros: This method returns a tensor filled with zeros that has a specified shape (此方法返回一个填充有具有指定形状的零的张量).
