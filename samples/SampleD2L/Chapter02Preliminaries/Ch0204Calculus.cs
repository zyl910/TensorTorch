using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.SampleD2L.Chapter02Preliminaries {
    internal class Ch0204Calculus {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

        public static void Output(TextWriter writer) {
            writer.WriteLine("### 2.4 Calculus");

            // 微积分
            // :label:sec_calculus
            // 
            // 在2500年前，古希腊人把一个多边形分成三角形，并把它们的面积相加，才找到计算多边形面积的方法。 为了求出曲线形状（比如圆）的面积，古希腊人在这样的形状上刻内接多边形。 如 :numref:fig_circle_area所示，内接多边形的等长边越多，就越接近圆。 这个过程也被称为逼近法（method of exhaustion）。
            // 
            // 用逼近法求圆的面积:label:fig_circle_area
            // 
            // 事实上，逼近法就是积分（integral calculus）的起源。 2000多年后，微积分的另一支，微分（differential calculus）被发明出来。 在微分学最重要的应用是优化问题，即考虑如何把事情做到最好。 正如在 :numref:subsec_norms_and_objectives中讨论的那样， 这种问题在深度学习中是无处不在的。
            // 
            // 在深度学习中，我们“训练”模型，不断更新它们，使它们在看到越来越多的数据时变得越来越好。 通常情况下，变得更好意味着最小化一个损失函数（loss function）， 即一个衡量“模型有多糟糕”这个问题的分数。 最终，我们真正关心的是生成一个模型，它能够在从未见过的数据上表现良好。 但“训练”模型只能将模型与我们实际能看到的数据相拟合。 因此，我们可以将拟合模型的任务分解为两个关键问题：
            // 
            // 优化（optimization）：用模型拟合观测数据的过程；
            // 泛化（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。
            // 为了帮助读者在后面的章节中更好地理解优化问题和方法， 本节提供了一个非常简短的入门教程，帮助读者快速掌握深度学习中常用的微分知识。
            // 
            // 导数和微分
            // 我们首先讨论导数的计算，这是几乎所有深度学习优化算法的关键步骤。 在深度学习中，我们通常选择对于模型参数可微的损失函数。 简而言之，对于每个参数， 如果我们把这个参数增加或减少一个无穷小的量，可以知道损失会以多快的速度增加或减少，
            // 
            // 假设我们有一个函数 𝑓:ℝ→ℝ
            //  ，其输入和输出都是标量。 (如果 𝑓
            //  的导数存在，这个极限被定义为)
            // 
            // (
            // 𝑓′(𝑥)=limℎ→0𝑓(𝑥+ℎ)−𝑓(𝑥)ℎ.
            //  
            // ) :eqlabel:eq_derivative
            // 
            // 如果 𝑓′(𝑎)
            //  存在，则称 𝑓
            //  在 𝑎
            //  处是可微（differentiable）的。 如果 𝑓
            //  在一个区间内的每个数上都是可微的，则此函数在此区间中是可微的。 我们可以将 :eqref:eq_derivative中的导数 𝑓′(𝑥)
            //  解释为 𝑓(𝑥)
            //  相对于 𝑥
            //  的瞬时（instantaneous）变化率。 所谓的瞬时变化率是基于 𝑥
            //  中的变化 ℎ
            //  ，且 ℎ
            //  接近 0
            //  。
            // 
            // 为了更好地解释导数，让我们做一个实验。 (定义 𝑢=𝑓(𝑥)=3𝑥2−4𝑥
            //  )如下：
            // 
            // %matplotlib inline
            // import numpy as np
            // from matplotlib_inline import backend_inline
            // from d2l import torch as d2l
            // ​
            // ​
            // def f(x):
            //     return 3 * x ** 2 - 4 * x
            // [通过令𝑥=1
            // 并让ℎ
            // 接近0
            // ，] :eqref:eq_derivative中(𝑓(𝑥+ℎ)−𝑓(𝑥)ℎ
            // 的数值结果接近2
            // )。 虽然这个实验不是一个数学证明，但稍后会看到，当𝑥=1
            // 时，导数𝑢′
            // 是2
            // 。
            // 
            // def numerical_lim(f, x, h):
            //     return (f(x + h) - f(x)) / h
            // ​
            // h = 0.1
            // for i in range(5):
            //     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
            //     h *= 0.1
            // h=0.10000, numerical limit=2.30000
            // h=0.01000, numerical limit=2.03000
            // h=0.00100, numerical limit=2.00300
            // h=0.00010, numerical limit=2.00030
            // h=0.00001, numerical limit=2.00003
            // 让我们熟悉一下导数的几个等价符号。 给定𝑦=𝑓(𝑥)
            // ，其中𝑥
            // 和𝑦
            // 分别是函数𝑓
            // 的自变量和因变量。以下表达式是等价的：
            // 
            // 𝑓′(𝑥)=𝑦′=𝑑𝑦𝑑𝑥=𝑑𝑓𝑑𝑥=𝑑𝑑𝑥𝑓(𝑥)=𝐷𝑓(𝑥)=𝐷𝑥𝑓(𝑥),
            // 
            // 其中符号𝑑𝑑𝑥
            // 和𝐷
            // 是微分运算符，表示微分操作。 我们可以使用以下规则来对常见函数求微分：
            // 
            // 𝐷𝐶=0
            // （𝐶
            // 是一个常数）
            // 𝐷𝑥𝑛=𝑛𝑥𝑛−1
            // （幂律（power rule），𝑛
            // 是任意实数）
            // 𝐷𝑒𝑥=𝑒𝑥
            // 𝐷ln(𝑥)=1/𝑥
            // 为了微分一个由一些常见函数组成的函数，下面的一些法则方便使用。 假设函数𝑓
            // 和𝑔
            // 都是可微的，𝐶
            // 是一个常数，则：
            // 
            // 常数相乘法则
            // 𝑑𝑑𝑥[𝐶𝑓(𝑥)]=𝐶𝑑𝑑𝑥𝑓(𝑥),
            // 
            // 加法法则
            // 
            // 𝑑𝑑𝑥[𝑓(𝑥)+𝑔(𝑥)]=𝑑𝑑𝑥𝑓(𝑥)+𝑑𝑑𝑥𝑔(𝑥),
            // 
            // 乘法法则
            // 
            // 𝑑𝑑𝑥[𝑓(𝑥)𝑔(𝑥)]=𝑓(𝑥)𝑑𝑑𝑥[𝑔(𝑥)]+𝑔(𝑥)𝑑𝑑𝑥[𝑓(𝑥)],
            // 
            // 除法法则
            // 
            // 𝑑𝑑𝑥[𝑓(𝑥)𝑔(𝑥)]=𝑔(𝑥)𝑑𝑑𝑥[𝑓(𝑥)]−𝑓(𝑥)𝑑𝑑𝑥[𝑔(𝑥)][𝑔(𝑥)]2.
            // 
            // 现在我们可以应用上述几个法则来计算𝑢′=𝑓′(𝑥)=3𝑑𝑑𝑥𝑥2−4𝑑𝑑𝑥𝑥=6𝑥−4
            // 。 令𝑥=1
            // ，我们有𝑢′=2
            // ：在这个实验中，数值结果接近2
            // ， 这一点得到了在本节前面的实验的支持。 当𝑥=1
            // 时，此导数也是曲线𝑢=𝑓(𝑥)
            // 切线的斜率。
            // 
            // [为了对导数的这种解释进行可视化，我们将使用matplotlib]， 这是一个Python中流行的绘图库。 要配置matplotlib生成图形的属性，我们需要(定义几个函数)。 在下面，use_svg_display函数指定matplotlib软件包输出svg图表以获得更清晰的图像。
            // 
            // 注意，注释#@save是一个特殊的标记，会将对应的函数、类或语句保存在d2l包中。 因此，以后无须重新定义就可以直接调用它们（例如，d2l.use_svg_display()）。
            // 
            // def use_svg_display():  #@save
            //     """使用svg格式在Jupyter中显示绘图"""
            //     backend_inline.set_matplotlib_formats('svg')
            // 我们定义set_figsize函数来设置图表大小。 注意，这里可以直接使用d2l.plt，因为导入语句 from matplotlib import pyplot as plt已标记为保存到d2l包中。
            // 
            // def set_figsize(figsize=(3.5, 2.5)):  #@save
            //     """设置matplotlib的图表大小"""
            //     use_svg_display()
            //     d2l.plt.rcParams['figure.figsize'] = figsize
            // 下面的set_axes函数用于设置由matplotlib生成图表的轴的属性。
            // 
            // #@save
            // def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
            //     """设置matplotlib的轴"""
            //     axes.set_xlabel(xlabel)
            //     axes.set_ylabel(ylabel)
            //     axes.set_xscale(xscale)
            //     axes.set_yscale(yscale)
            //     axes.set_xlim(xlim)
            //     axes.set_ylim(ylim)
            //     if legend:
            //         axes.legend(legend)
            //     axes.grid()
            // 通过这三个用于图形配置的函数，定义一个plot函数来简洁地绘制多条曲线， 因为我们需要在整个书中可视化许多曲线。
            // 
            // #@save
            // def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
            //          ylim=None, xscale='linear', yscale='linear',
            //          fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
            //     """绘制数据点"""
            //     if legend is None:
            //         legend = []
            // ​
            //     set_figsize(figsize)
            //     axes = axes if axes else d2l.plt.gca()
            // ​
            //     # 如果X有一个轴，输出True
            //     def has_one_axis(X):
            //         return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
            //                 and not hasattr(X[0], "__len__"))
            // ​
            //     if has_one_axis(X):
            //         X = [X]
            //     if Y is None:
            //         X, Y = [[]] * len(X), X
            //     elif has_one_axis(Y):
            //         Y = [Y]
            //     if len(X) != len(Y):
            //         X = X * len(Y)
            //     axes.cla()
            //     for x, y, fmt in zip(X, Y, fmts):
            //         if len(x):
            //             axes.plot(x, y, fmt)
            //         else:
            //             axes.plot(y, fmt)
            //     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
            // 现在我们可以[绘制函数𝑢=𝑓(𝑥)
            // 及其在𝑥=1
            // 处的切线𝑦=2𝑥−3
            // ]， 其中系数2
            // 是切线的斜率。
            // 
            // x = np.arange(0, 3, 0.1)
            // plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
            // <Figure size 350x250 with 1 Axes>
            // 偏导数
            // 到目前为止，我们只讨论了仅含一个变量的函数的微分。 在深度学习中，函数通常依赖于许多变量。 因此，我们需要将微分的思想推广到多元函数（multivariate function）上。
            // 
            // 设𝑦=𝑓(𝑥1,𝑥2,…,𝑥𝑛)
            // 是一个具有𝑛
            // 个变量的函数。 𝑦
            // 关于第𝑖
            // 个参数𝑥𝑖
            // 的偏导数（partial derivative）为：
            // 
            // ∂𝑦∂𝑥𝑖=limℎ→0𝑓(𝑥1,…,𝑥𝑖−1,𝑥𝑖+ℎ,𝑥𝑖+1,…,𝑥𝑛)−𝑓(𝑥1,…,𝑥𝑖,…,𝑥𝑛)ℎ.
            // 
            // 为了计算∂𝑦∂𝑥𝑖
            // ， 我们可以简单地将𝑥1,…,𝑥𝑖−1,𝑥𝑖+1,…,𝑥𝑛
            // 看作常数， 并计算𝑦
            // 关于𝑥𝑖
            // 的导数。 对于偏导数的表示，以下是等价的：
            // 
            // ∂𝑦∂𝑥𝑖=∂𝑓∂𝑥𝑖=𝑓𝑥𝑖=𝑓𝑖=𝐷𝑖𝑓=𝐷𝑥𝑖𝑓.
            // 
            // 梯度
            // :label:subsec_calculus-grad
            // 
            // 我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的梯度（gradient）向量。 具体而言，设函数𝑓:ℝ𝑛→ℝ
            // 的输入是 一个𝑛
            // 维向量𝐱=[𝑥1,𝑥2,…,𝑥𝑛]⊤
            // ，并且输出是一个标量。 函数𝑓(𝐱)
            // 相对于𝐱
            // 的梯度是一个包含𝑛
            // 个偏导数的向量:
            // 
            // ∇𝐱𝑓(𝐱)=[∂𝑓(𝐱)∂𝑥1,∂𝑓(𝐱)∂𝑥2,…,∂𝑓(𝐱)∂𝑥𝑛]⊤,
            // 
            // 其中∇𝐱𝑓(𝐱)
            // 通常在没有歧义时被∇𝑓(𝐱)
            // 取代。
            // 
            // 假设𝐱
            // 为𝑛
            // 维向量，在微分多元函数时经常使用以下规则:
            // 
            // 对于所有𝐀∈ℝ𝑚×𝑛
            // ，都有∇𝐱𝐀𝐱=𝐀⊤
            // 对于所有𝐀∈ℝ𝑛×𝑚
            // ，都有∇𝐱𝐱⊤𝐀=𝐀
            // 对于所有𝐀∈ℝ𝑛×𝑛
            // ，都有∇𝐱𝐱⊤𝐀𝐱=(𝐀+𝐀⊤)𝐱
            // ∇𝐱‖𝐱‖2=∇𝐱𝐱⊤𝐱=2𝐱
            // 同样，对于任何矩阵𝐗
            // ，都有∇𝐗‖𝐗‖2𝐹=2𝐗
            // 。 正如我们之后将看到的，梯度对于设计深度学习中的优化算法有很大用处。
            // 
            // 链式法则
            // 然而，上面方法可能很难找到梯度。 这是因为在深度学习中，多元函数通常是复合（composite）的， 所以难以应用上述任何规则来微分这些函数。 幸运的是，链式法则可以被用来微分复合函数。
            // 
            // 让我们先考虑单变量函数。假设函数𝑦=𝑓(𝑢)
            // 和𝑢=𝑔(𝑥)
            // 都是可微的，根据链式法则：
            // 
            // 𝑑𝑦𝑑𝑥=𝑑𝑦𝑑𝑢𝑑𝑢𝑑𝑥.
            // 
            // 现在考虑一个更一般的场景，即函数具有任意数量的变量的情况。 假设可微分函数𝑦
            // 有变量𝑢1,𝑢2,…,𝑢𝑚
            // ，其中每个可微分函数𝑢𝑖
            // 都有变量𝑥1,𝑥2,…,𝑥𝑛
            // 。 注意，𝑦
            // 是𝑥1,𝑥2，…,𝑥𝑛
            // 的函数。 对于任意𝑖=1,2,…,𝑛
            // ，链式法则给出：
            // 
            // ∂𝑦∂𝑥𝑖=∂𝑦∂𝑢1∂𝑢1∂𝑥𝑖+∂𝑦∂𝑢2∂𝑢2∂𝑥𝑖+⋯+∂𝑦∂𝑢𝑚∂𝑢𝑚∂𝑥𝑖
            // 
            // 小结
            // 微分和积分是微积分的两个分支，前者可以应用于深度学习中的优化问题。
            // 导数可以被解释为函数相对于其变量的瞬时变化率，它也是函数曲线的切线的斜率。
            // 梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数。
            // 链式法则可以用来微分复合函数。

            // done.
            writer.WriteLine();
        }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    }
}
