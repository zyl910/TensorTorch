﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.SampleD2L.Chapter02Preliminaries {
    internal class Ch0202Pandas {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

        public static void Output(TextWriter writer) {
            writer.WriteLine("### 2.2 Pandas");

            // 数据预处理
            // :label:sec_pandas
            // 
            // 为了能用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始， 而不是从那些准备好的张量格式数据开始。 在Python中常用的数据分析工具中，我们通常使用pandas软件包。 像庞大的Python生态系统中的许多其他扩展包一样，pandas可以与张量兼容。 本节我们将简要介绍使用pandas预处理原始数据，并将原始数据转换为张量格式的步骤。 后面的章节将介绍更多的数据预处理技术。
            // 
            // 读取数据集
            // 举一个例子，我们首先(创建一个人工数据集，并存储在CSV（逗号分隔值）文件) ../data/house_tiny.csv中。 以其他格式存储的数据也可以通过类似的方式进行处理。 下面我们将数据集按行写入CSV文件中。
            // 
            // import os
            // ​
            // os.makedirs(os.path.join('..', 'data'), exist_ok=True)
            // data_file = os.path.join('..', 'data', 'house_tiny.csv')
            // with open(data_file, 'w') as f:
            //     f.write('NumRooms,Alley,Price\n')  # 列名
            //     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
            //     f.write('2,NA,106000\n')
            //     f.write('4,NA,178100\n')
            //     f.write('NA,NA,140000\n')
            // 要[从创建的CSV文件中加载原始数据集]，我们导入pandas包并调用read_csv函数。该数据集有四行三列。其中每行描述了房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）。
            // 
            // # 如果没有安装pandas，只需取消对以下行的注释来安装pandas
            // # !pip install pandas
            // import pandas as pd
            // ​
            // data = pd.read_csv(data_file)
            // print(data)
            //    NumRooms Alley   Price
            // 0       NaN  Pave  127500
            // 1       2.0   NaN  106000
            // 2       4.0   NaN  178100
            // 3       NaN   NaN  140000
            // 处理缺失值
            // 注意，“NaN”项代表缺失值。 [为了处理缺失的数据，典型的方法包括插值法和删除法，] 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。 在(这里，我们将考虑插值法)。
            // 
            // 通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
            // 
            // inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
            // inputs = inputs.fillna(inputs.mean())
            // print(inputs)
            //    NumRooms Alley
            // 0       3.0  Pave
            // 1       2.0   NaN
            // 2       4.0   NaN
            // 3       3.0   NaN
            // [对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。] 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
            // 
            // inputs = pd.get_dummies(inputs, dummy_na=True)
            // print(inputs)
            //    NumRooms  Alley_Pave  Alley_nan
            // 0       3.0           1          0
            // 1       2.0           0          1
            // 2       4.0           0          1
            // 3       3.0           0          1
            // 转换为张量格式
            // [现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。] 当数据采用张量格式后，可以通过在 :numref:sec_ndarray中引入的那些张量函数来进一步操作。
            // 
            // import torch
            // ​
            // X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
            // X, y
            // (tensor([[3., 1., 0.],
            //          [2., 0., 1.],
            //          [4., 0., 1.],
            //          [3., 0., 1.]], dtype=torch.float64),
            //  tensor([127500, 106000, 178100, 140000]))
            // 小结
            // pandas软件包是Python中常用的数据分析工具中，pandas可以与张量兼容。
            // 用pandas处理缺失的数据时，我们可根据情况选择用插值法和删除法。

            // done.
            writer.WriteLine();
        }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    }
}
