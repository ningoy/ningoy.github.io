---
title: 使用ARIMA模型进行时间序列分析
date: 2021-06-04 17:54:00
tags:
- 时序预测
categories:
- 时序预测
---



# ARIMA模型

ARIMA模型，自回归移动平均模型（ARIMA，Autoregressive Integrated Moving Average Model），是统计模型中最常见的一种用来进行时间序列预测的模型，旨在描绘数据的自回归性（autocorrelations）。如果观测值并非彼此独立，一个观测值可能会在i个时间单位后与另一个观测值相关，形成一种称为自相关的关系。自相关可以削减基于时间的预测模型（例如时间序列图）的准确性，并导致数据的错误解释。

在引入ARIMA模型之前，我们需要先讨论平稳性（stationarity）和差分时间序列（differencing time series）的相关知识。

## 平稳性和差分

平稳的时间序列的性质不随观测时间的变化而变化。因此具有趋势或季节性的时间序列不是平稳时间序列——趋势和季节性使得时间序列在不同时段呈现不同特质。与他们相反，白噪声序列（white noise series）则是平稳的——不管观测的时间如何变化，它看起来都应该是一样的。

在判断平稳性上，下面这个例子容易让人混淆：如果一个循环变化的时间序列没有趋势和季节性，那么它仍然是平稳的。这是因为这些循环变化并没有一个固定的周期，因此在进行观测之前我们无法知道循环变化的峰值和谷值会出现在哪个位置。

一般而言，一个平稳的时间序列从长期来看不存在可预测的特征。它的时间曲线图（time plots）反映出这个序列近似于水平（尽管可能存在一些周期性的变化）并保持固定的方差。

![图中的时间序列有哪些是平稳的？（a）连续292天的谷歌股价; （b）连续292天谷歌股价的每日变化量; （c）美国各年的罢工总次数; （d）美国独立家庭住宅的每月价格; （e）按不变美元计算的美国的鸡蛋价格; （e）每月在澳大利亚维多利亚州被屠宰的猪的数量; （g）每年在加拿大西北的麦肯齐河停留的猞猁数量; （h）澳大利亚每月啤酒产量; （i）澳大利亚每月发电量](http://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/stationary-1.png?x-oss-process=PicGo)

> 图 8.1: 图中的时间序列有哪些是平稳的？（a）连续292天的谷歌股价; （b）连续292天谷歌股价的每日变化量; （c）美国各年的罢工总次数; （d）美国独立家庭住宅的每月价格; （e）按不变美元计算的美国的鸡蛋价格; （e）每月在澳大利亚维多利亚州被屠宰的猪的数量; （g）每年在加拿大西北的麦肯齐河停留的猞猁数量; （h）澳大利亚每月啤酒产量; （i）澳大利亚每月发电量

考虑图[8.1](http://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/stationary-1.png?x-oss-process=PicGo)中的九个时间序列，其中有哪些是平稳的时间序列？

显然存在季节性的序列（d）、（h）和（i）可以被排除。存在趋势的序列（a）、（c）、（e）、（f）和（i）也应该被排除，除此之外，序列（i）的方差随时间增大，也不符合平稳时间序列的性质。用上述方法排除后，剩下的（b）和（g）是平稳时间序列。

序列（g）的循环变化让它第一眼看上去不太平稳，但是这种变化其实是不定期的——当猞猁的数量超过食物承载的上限时，它们会停止繁殖从而使得数量回落到非常低的水平，之后食物来源的再生使得猞猁数量重新增长，周而复始。从长期来看，这种循环的时间点是不能预测的，因此序列（g）是平稳的。

### 差分

在图 [8.1](http://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/stationary-1.png?x-oss-process=PicGo)中，我们注意到（a）中谷歌股价数并不平稳，但（b）中谷歌股价每天的变化量则是平稳的。这向我们展示了一种让非平稳时间序列变平稳的方法——计算相邻观测值之间的差值，这种方法被称为**差分**。

诸如对数变换的变换方法可用于平稳化（stabilize）时间序列的方差。差分则可以通过去除时间序列中的一些变化特征来平稳化它的均值，并因此消除（或减小）时间序列的趋势和季节性。

和时间曲线图一样，自相关图（ACF图）也能帮助我们识别非平稳时间序列。 对于一个平稳时间序列，自相关系数（ACF）会快速的下降到接近 0 的水平，然而非平稳时间序列的自相关系数会下降的比较缓慢。同样的，非平稳时间序列的 $r_1$ 通常非常大并且为正值。

<img src="https://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/20210606013141.png?x-oss-process=PicGo" alt="谷歌股价（左图）和谷歌股价的每日变化（右图）的自相关系数。" style="zoom:67%;" />

> 图 8.2: 谷歌股价（左图）和谷歌股价的每日变化（右图）的自相关系数。

```python
import yfinance as yf
import statsmodels.api as sm
goog = yf.download("GOOG", start="2013-01-01", end="2013-12-06")
sm.stats.acorr_ljungbox(goog['Close'] - goog['Open'], lags=[10], return_df=True)

# 		lb_stat		lb_pvalue
# 10	10.498818	0.397872
```

差分后的谷歌股价的自相关图看起来像白噪声序列。所有自回归系数都在95%的置信度以内，并且 Ljung-Box 检验中 $Q^*$ 统计量的*p*值为 0.355 (for $h=10$)。这反映出谷歌股价的*每日变化*在本质上是一个与过去时间无关的随机值。


### 随机游走模型

差分序列是指原序列的连续观测值之间的*变化值*组成的时间序列，它可以被表示为：
$$
y'_t = y_t - y_{t-1}.
$$
差分序列的长度为 $T-1$，因为$t=1$时，公式中的差值无法计算。

当差分序列是白噪声时，原序列的模型可以表示为：
$$
y_t - y_{t-1} = \varepsilon_t,
$$
这里的$\varepsilon_t$为白噪声。调整上式，即可得到“随机游走”模型：
$$
y_t = y_{t-1} + \varepsilon_t.
$$
随机游走模型在非平稳时间序列数据中应用广泛，特别是金融和经济数据。典型的随机游走通常具有以下特征：

- 长期的明显上升或下降趋势。
- 游走方向上突然的、不能预测的变化。

由于未来变化是不可预测的，随机游走模型的预测值为上一次观测值，并且其上升和下降的可能性相同。因此，随机游走模型适用于朴素（naive）的预测。

通过稍许改进，我们可以让差值均值不为零。从而：
$$
y_t - y_{t-1} = c + \varepsilon_t\quad\text{or}\quad {y_t = c + y_{t-1} + \varepsilon_t}\: .
$$
$c$值是连续观测值变化的平均值。如果$c$值为正，则之前的平均变化情况是增长的，因此$y_t$将倾向于继续向上漂移（drift）。反之如果$c$值为负，$y_t$将倾向于向下漂移。

### 二阶差分

有时差分后的数据仍然不平稳，所以可能需要再一次对数据进行差分来得到一个平稳的序列：
$$
\begin{align*}
  y''_{t}  &=  y'_{t}  - y'_{t - 1} \\
           &= (y_t - y_{t-1}) - (y_{t-1}-y_{t-2})\\
           &= y_t - 2y_{t-1} +y_{t-2}.
\end{align*}
$$
在这种情况下，序列$y_t''$的长度为$T-2$。之后我们可以对原数据的“变化的变化”进行建模。在现实应用中，通常没有必要进行二阶以上的差分。

## 季节性差分

季节性差分是对一个观测值和相对应的前一年的观测值之间进行差分。因此有：
$$
y_t' = y_t - y_{t-m},
$$
其中$m=$一年中的季节数量。这也被称为“延迟-$m$差值”，因为相减的两个观测值之间的时间间隔为$m$。

如果季节性差分数据是白噪声，则原数据可以用一个合适的模型来拟合：
$$
y_t = y_{t-m}+\varepsilon_t.
$$
这个模型的预测值等于对应季节的上一次观测值。换言之，这个模型提供季节性的朴素（naive）预测。

图[8.3](https://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/20210606222125.png?x-oss-process=PicGo) 中下方的图显示的是 A10（抗糖尿病）药剂在澳大利亚月销售量的对数的季节差值。经过变换和差分，序列变得相对平稳。

```R
cbind("销售量 ($百万)" = a10,
      "每月销量对数" = log(a10),
      "每年销量变化对数" = diff(log(a10),12)) %>%
  autoplot(facets=TRUE) +
    xlab("年份") + ylab("") +
    ggtitle("抗糖尿病药剂销量")+
  theme(text = element_text(family = "STHeiti"))+
  theme(plot.title = element_text(hjust = 0.5))
```

![A10（抗糖尿病）药剂销量的对数和季节性差值数据，对数变换稳定了方差，而季节性差分去除了数据的趋势和季节性。](https://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/20210606222125.png?x-oss-process=PicGo)

> 图 8.3: A10（抗糖尿病）药剂销量的对数和季节性差值数据，对数变换稳定了方差，而季节性差分去除了数据的趋势和季节性。

为了区别季节差分和一般的差分，我们有时将一般的差分称为“一步差分”，即差值的延迟期数为 1。

正如图[8.4](https://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/20210606222427.png?x-oss-process=PicGo)所示，我们有时会同时使用季节性差分和一般的差分方法来得到平稳时间序列。在图中，我们先对数据进行对数变换（第二幅图），之后进行季节性差分（第三幅图）。经过上述操作后的数据仍然看起来有点非平稳，所以我们又进行了一次差分（第四幅图）。

```R
cbind("十亿千瓦时" = usmelec,
      "对数" = log(usmelec),
      "季节性\n 差分对数" = diff(log(usmelec),12),
      "二次\n 差分对数" = diff(diff(log(usmelec),12),1)) %>%
  autoplot(facets=TRUE) +
    xlab("年份") + ylab("") +
    ggtitle("美国电网每月发电量")+
  theme(text = element_text(family = "STHeiti"))+
  theme(plot.title = element_text(hjust = 0.5))
```

![第一幅图：美国电网每月发电量 (十亿千瓦时)。其他图显示的是该数据经过不同的变换和差分后的情况。](https://wxbfans-ink.oss-cn-beijing.aliyuncs.com/img/20210606222427.png?x-oss-process=PicGo)

> 图 8.4: 第一幅图：美国电网每月发电量 (十亿千瓦时)。其他图显示的是该数据经过不同的变换和差分后的情况。

选择使用哪些差分方式具有一定的主观性。图[8.3](https://otexts.com/fppcn/stationarity.html#fig:a10diff)中季节性差分的数据看起来和图[8.4](https://otexts.com/fppcn/stationarity.html#fig:usmelec)中季节性差分的数据差异并不大。在后一种情况中，我们可能会使用季节性差分后的数据，而不是进一步对数据进行差分。在前一种情况中，我们也可能认为季节性差分后的数据仍然不够平稳，因而进一步进行差分。我们将在后文中讨论一些严谨的差分检验，然而选择使用何种方式仍然是一个主观选择的过程，不同的分析师可能会做出不同的选择。

假如用$y'_t = y_t - y_{t-m}$表示季节性的差分序列，那么它的二阶差分序列则为：
$$
\begin{align*}
y''_t &= y'_t - y'_{t-1} \\
      &= (y_t - y_{t-m}) - (y_{t-1} - y_{t-m-1}) \\
      &= y_t -y_{t-1} - y_{t-m} + y_{t-m-1}\:
\end{align*}
$$
当季节性差值和第一差值都被使用时，两者的先后顺序并不会影响结果——变换顺序后的结果仍是一样的。然而，如果数据的季节性特征比较强，我们建议先进行季节性差分，因为有时经过季节性差分的数据已经足够平稳，没有必要进行后续的差分。如果先进行第一差分，我们仍将需要做一次季节性差分。

当我们使用差分时，有一点非常重要：差值应该是可解释（interpretable）的。第一差分是相邻观测值之间的差值，季节性差分是相邻年份的观测值的变化。其他延迟期数的差分很难和这两者一样易于解释，因此应该尽力避免。

### 单位根检验

*单位根检验*是一种更客观的判定是否需要差分的方法。这个针对平稳性的统计假设检验被用于判断是否需要差分方法来让数据更平稳。

单位根检验的方法有很多种，它们基于不同的假设，因此可能产生相互矛盾的结果。在我们的分析中，采用 *Kwiatkowski-Phillips-Schmidt-Shin (KPSS)* 检验(Kwiatkowski, Phillips, Schmidt, & Shin, [1992](https://otexts.com/fppcn/stationarity.html#ref-KPSS92))。在此检验中，原假设为数据是平稳的，我们要寻找能够证明原假设是错误的证据。因此，很小的P值（例如小于0.05）说明需要进行差分。该检验可以使用程序包[**statsmodels**](https://www.statsmodels.org/stable/index.html)中的 `statsmodels.tsa.stattools.kpss()` 函数进行计算。

例如，让我们对谷歌的股价数据进行该检验。

```R
from statsmodels.tsa.stattools import kpss
import pandas as pd
import yfinance as yf
data = yf.download("GOOG", start="2013-01-01", end="2013-12-06")
kpss_output = pd.Series(kpss(data['Close'], regression='c')[0:3], 
          index=['Test Statistic', 'p-value', 'Lags used'])
for key, value in kpss(data['Close'], regression='c')[3].items():
    kpss_output['Critical Value (%s)' % key] = value
kpss_output
# Test Statistic            1.280268
#
# p-value                   0.010000
# Lags used                15.000000
# Critical Value (10%)      0.347000
# Critical Value (5%)       0.463000
# Critical Value (2.5%)     0.574000
# Critical Value (1%)       0.739000
# dtype: float64
```

检验统计量的值远大于临界值 1%，可以拒绝原假设，也就是说，该序列不平稳。我们可以对数据进行差分，再次进行检验。

```python
from statsmodels.tsa.stattools import kpss
import pandas as pd
import yfinance as yf
data = yf.download("GOOG", start="2013-01-01", end="2013-12-06")
kpss_output = pd.Series(kpss(data['Close'] - data['Open'], regression='c')[0:3], 
          index=['Test Statistic', 'p-value', 'Lags used'])
for key, value in kpss(data['Close'] - data['Open'], regression='c')[3].items():
    kpss_output['Critical Value (%s)' % key] = value
kpss_output
# Test Statistic            0.088083
# 
# p-value                   0.100000
# Lags used                15.000000
# Critical Value (10%)      0.347000
# Critical Value (5%)       0.463000
# Critical Value (2.5%)     0.574000
# Critical Value (1%)       0.739000
# dtype: float64
```

这次检验的统计量的值很小，处在期望的范围以内，因此可以推断出差分后的序列是平稳的。

函数`ndiffs`可以通过一系列的KPSS检验来确定合适的一阶差分次数。

```R
ndiffs(goog)
#> [1] 1
```

从上面的 KPSS 检验可以看出，需要进行一次差分来让`goog`数据变得平稳。

与上述函数类似，`nsdiffs`函数可以用来确定是否需要进行季节性差分，它通过季节性强度来确定合适的季节性差分次数，如果$F_S<0.64$，不需要进行季节性差分，否则需要进行一次季节性差分。

对美国月度用电数据使用`nsdiffs()`函数。

```R
usmelec %>% log() %>% nsdiffs()
#> [1] 1
usmelec %>% log() %>% diff(lag=12) %>% ndiffs()
#> [1] 1
```

由于`nsdiffs()`函数返回 1，（说明需要进行一次季节性差分），我们对季节差分后的数据运行`ndiffs()`函数。这些函数的运行结果说明我们应该进行一次季节性差分和一次一步差分。

### 参考文献

Hyndman R J, Athanasopoulos G. Forecasting: principles and practice[M]. OTexts, 2018.

























ARIMA模型的优缺点，优点是模型十分简单，只需要内生变量而不需要借助其他外生变量。以最简单的模型 Y = aX + b 为例，X 是自变量，Y是因变量，a 和 b 都是外生变量，是由模型的外部因素决定的。

缺点：

1. 要求时序数据是稳定的（stationary），或者通过差分化（differencing）后是稳定的。
2. 本质上只能捕捉线性关系，而不能捕捉非线性关系。

那么要如何判断时序数据是稳定的呢？

一个时间序列的随机变量是稳定的，当且仅当它的统计特征都是独立于时间的（是关于时间的常量）。

1. 稳定的数据是没有趋势（trend），没有周期性（seasonality）的；即它的均值，在时间轴上拥有常量的振幅，并且它的方差，在时间轴上是趋于同一个稳定的值的。
2. 可以用Dickey-Fuller Test 进行假设检验。

ARIMA的参数与数学形式

ARIMA模型有三个参数p,d,q

+ p -- 代表预测模型中采用的时序数据本身的滞后数（lags），也叫做AR/Auto-Regressive项。
+ d -- 代表时序数据需要进行几阶差分化才是稳定的，也叫Integrated项。
+ q -- 代表预测模型中采用的预测误差的滞后数（lags），也叫MA/Moving Average项。

差分 -- 假设y表示t时刻的Y的差分

ARIMA的预测模型可以表示为：

Y的预测值 = 常量c and/or 一个或多个最近时间的Y的加权 and/or 一个或多个最近时间的预测误差。

假设p,q,d已知：

ARIMA用数学形式表示为:
$$
ytˆ=μ+ϕ1∗yt−1+...+ϕp∗yt−p+θ1∗et−1+...+θq∗et−q
其中,ϕ表示AR的系数，θ表示MA的系数
$$
ARIMA模型的几个特例

1. ARIMA(0,1,0) = random walk:

   当d=1,p 和 q 为0 时，叫做random walk，每一个时刻的Y，只与上一时刻的Y有关。
   $$
   Yˆt=μ+Yt−1
   $$

2. ARIMA(1,0,0) = first-order autoregressive model:

   当p=1,d=0,q=0,说明时序数据是稳定的和自相关的。一个时刻的Y值只与上一个时刻的Y值有关
   $$
   Yˆt=μ+ϕ1∗Yt−1.where, ϕ∈[−1,1],是一个斜率系数
   $$

3. ARIMA(1,1,0) = differenced first-order autoregressive model:
   p=1,d=1,q=0. 说明时序数据在一阶差分化之后是稳定的和自回归的。即一个时刻的差分（y）只与上一个时刻的差分有关。
   $$
   yˆt=μ+ϕ1∗yt−1结合一阶差分的定义，也可以表示为：Yˆt−Yt−1=μ+ϕ1∗(Yt−1−Yt−2)或者Yˆt=μ+Yt−1+ϕ1∗(Yt−1−Yt−2)
   $$
   
4. ARIMA(0,1,1) = simple exponential smoothing with growth.
    p=0, d=1 ,q=1.说明数据在一阶差分后市稳定的和移动平均的。即一个时刻的估计值的差分与上一个时刻的预测误差有关。
$$
  yˆt=μ+α1∗et−1注意q=1的差分yt与p=1的差分yt的是不一样的其中，yˆt=Yˆt−Yˆt−1, et−1=Yt−1−Yˆt−1,设θ1=1−α1则也可以写成：Yˆt=μ+Yˆt−1+α1(Yt−1−Yˆt−1)=μ+Yt−1−θ1∗et−1
$$

5. ARIMA(2,1,2)
   在通过上面的例子，可以很轻松的写出它的预测模型：
   $$
   yˆt=μ+ϕ1∗yt−1+ϕ2∗yt−2−θ1∗et−1−θ2∗et−2也可以写成:Yˆt=μ+ϕ1∗(Yt−1−Yt−2)+ϕ2∗(Yt−2−Yt−3)−θ1∗(Yt−1−Yˆt−1)−θ2∗(Yt−2−Yˆt−2)
   $$

6. ARIMA(2,2,2)
   $$
   yˆt=μ+ϕ1∗yt−1+ϕ2∗yt−2−θ1∗et−1−θ2∗et−2Yˆt=μ+ϕ1∗(Yt−1−2Yt−2+Yt−3)+ϕ2∗(Yt−2−2Yt−3+Yt−4)−θ1∗(Yt−1−Yˆt−1)−θ2∗(Yt−2−Yˆt−2)
   $$

ARIMA建模基本步骤

1. 获取被观测系统时间序列数据；
2. 对数据绘图，观测是否为平稳时间序列；对于非平稳时间序列要先进行d阶差分运算，化为平稳时间序列；
3. 经过第二步处理，已经得到平稳时间序列。要对平稳时间序列分别求得其自相关系数ACF 和偏自相关系数PACF，通过对自相关图和偏自相关图的分析，得到最佳的阶层 p 和阶数 q
4. 由以上得到的d、q、p，得到ARIMA模型。然后开始对得到的模型进行模型检验。

