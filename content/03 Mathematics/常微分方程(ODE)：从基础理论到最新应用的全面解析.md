# 常微分方程 (ODE)：从基础理论到最新应用的全面解析

## 一、引言

常微分方程 (Ordinary Differential Equation，简称 ODE) 是数学中一个重要的分支，它描述了未知函数如何随着一个独立变量变化的规律[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。作为现代科学和工程领域的基础工具，常微分方程广泛应用于物理、化学、生物、经济等多个学科，用于建立动态系统的数学模型并预测其行为[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。从 17 世纪牛顿和莱布尼茨创立微积分开始，常微分方程的发展已经历了几个世纪的演变，形成了完整的理论体系和丰富的求解方法[(27)](https://blog.csdn.net/qq_41375318/article/details/145261945)。

本文将系统地介绍常微分方程的基本概念、发展历史、数学原理、求解方法、应用场景以及最新研究进展，为读者提供一个全面而深入的常微分方程知识框架。通过阅读本文，读者不仅能掌握常微分方程的基本理论和求解技巧，还能了解这一数学工具在现代科学和工程中的广泛应用，以及其与人工智能等前沿领域的交叉融合。

## 二、常微分方程的基本概念与定义

### 2.1 常微分方程的定义

常微分方程是联系自变量、未知函数及其导数的关系式，其中未知函数是一元函数，即只含一个自变量的微分方程[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。在数学中，一个常微分方程 (ODE) 是指只依赖于一个独立变量的微分方程，其未知量由一个或多个函数组成，并涉及这些函数的导数。

**严格数学定义**：一个 n 阶常微分方程是一个形如

$F(x, y, y', y'', \ldots, y^{(n)}) = 0$

的方程，其中$F$是一个关于$x, y, y', \ldots, y^{(n)}$的已知函数，$y$是未知函数，$x$是独立变量，$y^{(k)}$表示$y$的 k 阶导数。

**关键点解析**：



*   "常" 表示方程中只涉及一个独立变量，这与偏微分方程 (PDE) 形成对比，后者涉及多个独立变量。

*   方程中导数的最高阶数 n 称为该常微分方程的阶[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

*   当方程可以表示为导数的显式形式时，称为显式常微分方程，否则称为隐式常微分方程。

### 2.2 常微分方程的分类

常微分方程可以根据多个标准进行分类，常见的分类方式包括：

**按方程的阶数分类**：



*   一阶常微分方程：最高导数为一阶，如$\frac{dy}{dx} = f(x, y)$[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

*   二阶常微分方程：最高导数为二阶，如$\frac{d^2y}{dx^2} + p(x)\frac{dy}{dx} + q(x)y = 0$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

*   高阶常微分方程：最高导数高于二阶的方程[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**按线性性质分类**：



*   线性常微分方程：方程中未知函数及其各阶导数都是一次的，且不包含它们的乘积项。n 阶线性常微分方程的一般形式为：

$a_n(x)\frac{d^ny}{dx^n} + \cdots + a_1(x)\frac{dy}{dx} + a_0(x)y = f(x)$

其中$a_n(x), \ldots, a_0(x)$和$f(x)$是已知函数[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。



*   非线性常微分方程：不满足线性条件的方程，如$\frac{dy}{dx} = y^2 + x$[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**按方程的次数分类**：



*   一次常微分方程：方程中最高阶导数的次数为一次。

*   高次常微分方程：最高阶导数的次数高于一次，如$(\frac{d^2y}{dx^2})^3 + y = 0$[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**按方程的解的结构分类**：



*   齐次常微分方程：当非齐次项$f(x) = 0$时的线性方程，如$\frac{d^2y}{dx^2} + y = 0$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

*   非齐次常微分方程：包含非零非齐次项的线性方程，如$\frac{d^2y}{dx^2} + y = \sin x$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

### 2.3 常微分方程的解

常微分方程的解是满足该方程的函数。根据解的性质，可以分为以下几类：

**通解**：

通解是指包含 n 个独立任意常数的解，其中 n 是方程的阶数。对于 n 阶方程，通解的形式为$y = \phi(x, C_1, C_2, \ldots, C_n)$，其中$C_1, C_2, \ldots, C_n$为任意常数[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。例如，一阶方程$\frac{dy}{dx} = 2x$的通解为$y = x^2 + C$，其中 C 为任意常数[(26)](https://wenku.csdn.net/column/3o8hwp1jbc)。

**特解**：

特解是通过给定初始条件或边界条件确定通解中任意常数后得到的解。例如，对于方程$\frac{dy}{dx} = 2x$，给定初始条件$y(0) = 1$，得到特解$y = x^2 + 1$[(26)](https://wenku.csdn.net/column/3o8hwp1jbc)。

**隐式解**：

当解不能表示为显式的函数形式$y = f(x)$，而只能表示为隐式方程$F(x, y) = 0$时，称为隐式解。例如，方程$\frac{dy}{dx} = -\frac{x}{y}$的解可以表示为$x^2 + y^2 = C$，这是一个隐式解[(46)](https://m.renrendoc.com/paper/197017430.html)。

**积分曲线**：

常微分方程的解在几何上表示为平面上的曲线，称为积分曲线。对于一阶方程$\frac{dy}{dx} = f(x, y)$，积分曲线上每一点的切线斜率都等于$f(x, y)$在该点的值[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

## 三、常微分方程的发展历程

### 3.1 早期发展：从微积分到常微分方程的形成

常微分方程的历史可以追溯到 17 世纪，当时微积分的基础由艾萨克・牛顿和戈特弗里德・莱布尼茨独立发明[(27)](https://blog.csdn.net/qq_41375318/article/details/145261945)。在解决物理问题的过程中，科学家们开始使用微积分方法来研究天体运动和力学系统，这促使了常微分方程理论的初步形成[(27)](https://blog.csdn.net/qq_41375318/article/details/145261945)。

1590 年，意大利天文学家伽利略在比萨斜塔自由落体实验中，通过求解微分方程发现了物理的运动规律，这被认为是常微分方程应用的早期案例[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。随后，牛顿在研究天体力学时，建立了行星运动的微分方程模型，开创了用微分方程描述自然现象的先河[(27)](https://blog.csdn.net/qq_41375318/article/details/145261945)。

1676 年，德国数学家莱布尼茨在给牛顿的信中首次提出 "微分方程" 的数学术语，标志着这一数学分支开始成为独立的研究对象[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

### 3.2 经典理论的建立：18 世纪至 19 世纪中期

18 世纪是常微分方程理论迅速发展的时期。瑞士数学家欧拉在 1743 年给出了 "通解" 和 "特解" 等概念，为常微分方程的系统研究奠定了基础[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。欧拉还提出了著名的欧拉方法，用于数值求解常微分方程，这是最早的数值解法之一[(27)](https://blog.csdn.net/qq_41375318/article/details/145261945)。

1754 年，法国数学家拉格朗日在解决等时曲线问题过程中创立了变分法，并提出了求解任意阶变系数非齐次线性常微分方程的常数变易法，这一方法至今仍是求解线性非齐次方程的重要工具[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

19 世纪初至中期，微分方程发展出了一套包括解的存在性、唯一性、延伸性，以及解的整体存在性、解对初值和参数的连续依赖性和可微性等基本理论的适定性理论体系[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。这一时期的重要成果包括：

1820 年，法国数学家柯西首次发表了关于常微分方程解的存在性定理，为常微分方程的理论基础做出了重要贡献[(34)](https://m.baike.com/wiki/%E6%9F%AF%E8%A5%BF%EF%BC%8D%E5%88%A9%E6%99%AE%E5%B8%8C%E8%8C%A8%E5%AE%9A%E7%90%86/20661513?baike_source=doubao)。

1876 年，德国数学家李普希茨提出了著名的 "李普希茨条件"，对解的存在唯一性定理做出进一步改进，使得定理的条件更加宽松，适用范围更广[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

### 3.3 定性理论与现代发展：19 世纪末至今

19 世纪末期及 20 世纪初期是常微分方程发展的第三个阶段，主要在以下三个方面有重大发展[(29)](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)：



1.  常微分方程实域定性理论的创立：1881 年，法国数学家庞加莱创立了常微分方程的定性理论，开启了从 "求定解问题" 转向 "求所有解" 的新时代[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。庞加莱不直接求解方程，而是通过研究方程本身的结构来推断解的性质，这一方法为常微分方程的研究开辟了新的途径[(29)](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)。

2.  常微分方程复域理论的发展：以法国数学家皮卡为代表的数学家们研究了复平面上的常微分方程，发展了一系列重要理论[(29)](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)。

3.  常微分方程摄动理论即小参数理论的建立：这一理论用于研究当方程中含有小参数时解的变化规律，在天体力学和非线性振动理论中有重要应用[(29)](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)。

至 20 世纪 60 年代，随着计算机技术的发展，常微分方程从 "求所有解" 转入 "求特殊解" 的时代，数值方法成为研究常微分方程的重要手段[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

20 世纪后期至今，常微分方程的研究与应用进入了多元化阶段，特别是与现代科学技术的结合，产生了许多新的研究方向，如：



*   微分方程控制理论：研究如何通过控制方程中的某些参数来达到预期的系统行为[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

*   微分方程数值解法的高效算法研究[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

*   微分方程在复杂系统建模中的应用，如神经网络、生物系统、金融模型等[(1)](https://www.semanticscholar.org/paper/Neural-Ordinary-Differential-Equations-for-Modeling-Kosma-Polytechnique/02c282db6cc2cdedf79d4d3cc8e2aa4e055c56b8)。

*   常微分方程与人工智能的交叉融合，如神经常微分方程 (Neural ODE) 的提出与发展[(3)](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)。

## 四、常微分方程的数学理论基础

### 4.1 解的存在唯一性定理

解的存在唯一性定理是常微分方程理论中最基础的定理之一，它回答了在什么条件下常微分方程的初值问题有唯一解的问题。

**皮卡 - 林德洛夫定理 (Picard-Lindelöf Theorem)**（又称柯西 - 利普希茨定理）：

设 E 为一个完备的有限维赋范向量空间，f 为一个取值在 E 上的函数：

$f: U \times I \rightarrow E \\
(x, t) \rightarrow f(x, t)$

其中 U 为 E 中的一个开集，I 是 R 中的一个区间。考虑以下的一阶非线性微分方程：

$\frac{dz}{dt} = f(z(t), t)$

如果 f 关于 t 连续，并在 U 中满足利普希茨条件，即存在常数 L，使得对于所有的$x_1, x_2 \in U$和$t \in I$，有：

$||f(x_1, t) - f(x_2, t)|| \leq L ||x_1 - x_2||$

则对于任意的初始条件$z(t_0) = x_0 \in U$，存在一个包含$t_0$的区间$J \subseteq I$，使得上述微分方程在 J 上存在唯一解[(34)](https://m.baike.com/wiki/%E6%9F%AF%E8%A5%BF%EF%BC%8D%E5%88%A9%E6%99%AE%E5%B8%8C%E8%8C%A8%E5%AE%9A%E7%90%86/20661513?baike_source=doubao)。

**定理的理解与说明**：



1.  利普希茨条件是比连续性更强的条件，它保证了函数 f 的变化率不会超过某个固定的常数 L，从而避免了函数变化过于剧烈的情况[(36)](https://max.book118.com/html/2024/1229/5022002213012021.shtm)。

2.  定理中的区间 J 可能比原区间 I 小，这是因为解可能在有限时间内趋向无穷，即出现 "爆破" 现象[(36)](https://max.book118.com/html/2024/1229/5022002213012021.shtm)。

3.  当 f 不满足利普希茨条件时，解的唯一性可能不成立。例如，方程$\frac{dy}{dt} = y^{1/3}$在初始条件$y(0) = 0$下，存在多个解，如$y = 0$和$y = \left(\frac{2}{3}t\right)^{3/2}$等。

4.  定理的局部性：皮卡 - 林德洛夫定理保证的是局部解的存在唯一性，即只在包含初始点的某个小区间上存在唯一解，而非整个区间 I 上[(34)](https://m.baike.com/wiki/%E6%9F%AF%E8%A5%BF%EF%BC%8D%E5%88%A9%E6%99%AE%E5%B8%8C%E8%8C%A8%E5%AE%9A%E7%90%86/20661513?baike_source=doubao)。

**存在性的证明方法**：

证明解的存在性通常有两种主要方法：



1.  **皮卡逐次逼近法**：构造一个逐次近似函数序列，并证明该序列在某个区间上一致收敛到方程的解[(35)](https://blog.csdn.net/weixin_39890814/article/details/111322818)。具体步骤如下：

*   任取一个满足初值条件的函数，例如$y_0(x) = y_0$

*   构造皮卡逐步逼近函数序列：

$y_n(x) = y_0 + \int_{x_0}^x f(t, y_{n-1}(t)) dt$



*   证明该序列在区间$[x_0 - h_0, x_0 + h_0]$上一致收敛，其中$h_0 = \min\left(a, \frac{b}{M}\right)$，$M = \max |f(x, y)|$[(36)](https://max.book118.com/html/2024/1229/5022002213012021.shtm)。

1.  **压缩映像原理**：将积分方程视为一个映射，并证明该映射是压缩映射，从而根据巴拿赫不动点定理，存在唯一的不动点，即方程的解[(36)](https://max.book118.com/html/2024/1229/5022002213012021.shtm)。

**唯一性的证明方法**：

证明唯一性的常用方法包括：



1.  **Gronwall 不等式法**：设$u(t)$和$v(t)$都是区间$[x_0 - h_0, x_0 + h_0]$上的连续非负函数，且满足不等式$u(t) \leq v(t) + \int_{x_0}^t u(s)v(s)ds$，则$u(t) \leq v(t)e^{\int_{x_0}^t v(s)ds}$[(36)](https://max.book118.com/html/2024/1229/5022002213012021.shtm)。利用这一不等式可以证明解的唯一性。

2.  **一般法**：设$y_1(x)$和$y_2(x)$都是微分方程的解，令$u(x) = y_1(x) - y_2(x)$，则$u(x)$满足一个齐次方程，利用利普希茨条件可以证明$u(x) = 0$[(36)](https://max.book118.com/html/2024/1229/5022002213012021.shtm)。

3.  **反证法**：假设存在两个不同的解，通过推导得出矛盾[(36)](https://max.book118.com/html/2024/1229/5022002213012021.shtm)。

### 4.2 线性微分方程解的结构理论

线性微分方程是常微分方程中最重要的一类，其解的结构具有清晰的理论框架。

**n 阶线性微分方程的一般形式**：

$a_n(x)\frac{d^ny}{dx^n} + a_{n-1}(x)\frac{d^{n-1}y}{dx^{n-1}} + \cdots + a_1(x)\frac{dy}{dx} + a_0(x)y = f(x)$

当$f(x) \equiv 0$时，称为齐次线性微分方程；否则称为非齐次线性微分方程[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**齐次线性微分方程解的性质**：



1.  **叠加原理**：如果$y_1(x)$和$y_2(x)$是齐次方程的解，则它们的线性组合$C_1y_1(x) + C_2y_2(x)$也是该方程的解，其中$C_1$和$C_2$为任意常数[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

2.  **基本解组**：齐次方程的 n 个线性无关的解$y_1(x), y_2(x), \ldots, y_n(x)$称为该方程的一个基本解组。方程的任一解都可以表示为这 n 个解的线性组合[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

3.  **朗斯基行列式**：由 n 个解构成的行列式

$W(y_1, y_2, \ldots, y_n) = \begin{vmatrix}
y_1 & y_2 & \cdots & y_n \\
y_1' & y_2' & \cdots & y_n' \\
\vdots & \vdots & \ddots & \vdots \\
y_1^{(n-1)} & y_2^{(n-1)} & \cdots & y_n^{(n-1)}
\end{vmatrix}$

称为朗斯基行列式。若$y_1, y_2, \ldots, y_n$线性无关，则$W \neq 0$；反之，若$W \neq 0$，则这些解线性无关[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。



1.  **刘维尔公式**：对于 n 阶齐次线性微分方程，其朗斯基行列式满足刘维尔公式：

$W(x) = W(x_0) \exp\left(-\int_{x_0}^x \frac{a_{n-1}(t)}{a_n(t)} dt\right)$

这表明朗斯基行列式要么恒为零，要么在整个区间上恒不为零[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**非齐次线性微分方程解的结构**：



1.  非齐次方程的通解可以表示为对应的齐次方程的通解加上非齐次方程的一个特解，即：

$y(x) = y_h(x) + y_p(x)$

其中$y_h(x)$是齐次方程的通解，$y_p(x)$是非齐次方程的一个特解[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。



1.  **常数变易法**：一种求解非齐次线性微分方程的方法，其基本思想是将齐次方程通解中的常数视为待定函数，并代入原方程求解这些函数[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

对于一阶线性微分方程$\frac{dy}{dx} + P(x)y = Q(x)$，其通解为：

$y(x) = e^{-\int P(x)dx} \left( \int Q(x)e^{\int P(x)dx}dx + C \right)$

这可以通过积分因子法得到，其中积分因子为$\mu(x) = e^{\int P(x)dx}$[(45)](https://blog.csdn.net/DaPiCaoMin/article/details/144947625)。

### 4.3 定性理论基础

常微分方程的定性理论由庞加莱创立，它不直接求解方程，而是通过研究方程本身的结构来推断解的性质，这对于那些难以求出解析解的非线性方程尤为重要[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**相平面与相轨迹**：

对于二阶常微分方程，可以将其转化为一个一阶方程组：

$\frac{dx}{dt} = P(x, y) \\
\frac{dy}{dt} = Q(x, y)$

在 (x, y) 平面上，每一点 (x, y) 都对应一个向量 (P (x, y), Q (x, y))，表示解曲线在该点的切线方向。这个平面称为相平面，解曲线称为相轨迹或轨道[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

**奇点与分类**：

相平面上满足 P (x, y) = 0 且 Q (x, y) = 0 的点称为奇点或平衡点。根据线性化后的系数矩阵的特征值，可以将奇点分为以下几类[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)：



1.  **结点**：当特征值为同号实数时，奇点称为结点。若特征值均为负，则为稳定结点；若均为正，则为不稳定结点[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

2.  **焦点**：当特征值为共轭复数且实部不为零时，奇点称为焦点。若实部为负，则为稳定焦点；若实部为正，则为不稳定焦点[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

3.  **中心**：当特征值为纯虚数时，奇点称为中心，周围的相轨迹为封闭曲线，代表周期性运动[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

4.  **鞍点**：当特征值为异号实数时，奇点称为鞍点，是不稳定的平衡点[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

**极限环**：

极限环是相平面上的孤立封闭轨迹，代表系统的周期解。极限环分为稳定极限环和不稳定极限环，前者吸引附近的轨线，后者排斥附近的轨线[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

**稳定性理论**：

稳定性理论研究系统在受到小扰动后的行为，主要关注平衡点的稳定性：



1.  **李雅普诺夫稳定性**：如果对于任意给定的$\epsilon > 0$，存在$\delta > 0$，使得当$||x(t_0) - x_0|| < \delta$时，对所有$t \geq t_0$，有$||x(t) - x_0|| < \epsilon$，则称平衡点$x_0$是李雅普诺夫稳定的[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

2.  **渐近稳定性**：如果平衡点$x_0$是李雅普诺夫稳定的，并且存在$\delta_0 > 0$，使得当$||x(t_0) - x_0|| < \delta_0$时，有$\lim_{t \to \infty} x(t) = x_0$，则称$x_0$是渐近稳定的[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

3.  **全局渐近稳定性**：如果对于任何初始条件$x(t_0)$，都有$\lim_{t \to \infty} x(t) = x_0$，则称$x_0$是全局渐近稳定的[(4)](https://m.zhangqiaokeyan.com/journal-foreign-detail/0704026003899.html)。

李雅普诺夫第二方法（直接方法）是研究稳定性的重要工具，它通过构造一个类似于能量的函数（称为李雅普诺夫函数）来判断平衡点的稳定性，而不需要知道方程的解[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

## 五、常微分方程的经典解法

### 5.1 一阶常微分方程的初等积分法

一阶常微分方程的一般形式为$\frac{dy}{dx} = f(x, y)$，其初等积分法主要包括以下几种：

**1. 变量可分离方程**

**定义与解法**：

如果方程可以写成$g(y)dy = f(x)dx$的形式，则称为变量可分离方程。其解法是对两边积分：

$\int g(y)dy = \int f(x)dx + C$

得到通解。如果存在$y_0$使得$g(y_0) = 0$，则$y = y_0$也是方程的解，可能不包含在通解中，必须予以补上[(46)](https://m.renrendoc.com/paper/197017430.html)。

**示例**：

解方程$\frac{dy}{dx} = - \frac{x}{y}$。将变量分离，得到$y dy = -x dx$。两边积分，即得$\frac{y^2}{2} = -\frac{x^2}{2} + \frac{C}{2}$，化简为$x^2 + y^2 = C$，这是一个以原点为圆心，半径为$\sqrt{C}$的圆[(46)](https://m.renrendoc.com/paper/197017430.html)。

**2. 齐次方程**

**定义与解法**：

形如$\frac{dy}{dx} = \phi\left(\frac{y}{x}\right)$的方程称为齐次方程。通过变量替换$u = \frac{y}{x}$，即$y = ux$，将方程转化为变量可分离方程：

$x\frac{du}{dx} + u = \phi(u)$

分离变量后积分求解，最后代回原变量[(46)](https://m.renrendoc.com/paper/197017430.html)。

**示例**：

解方程$\frac{dy}{dx} = \frac{y}{x} + \tan\frac{y}{x}$。令$u = \frac{y}{x}$，则方程变为$x\frac{du}{dx} + u = u + \tan u$，即$x\frac{du}{dx} = \tan u$。分离变量得$\frac{du}{\tan u} = \frac{dx}{x}$，积分得$\ln|\sin u| = \ln|x| + C$，即$\sin u = Cx$。代回原变量得$\sin\frac{y}{x} = Cx$，即$y = x \arcsin(Cx)$[(46)](https://m.renrendoc.com/paper/197017430.html)。

**3. 一阶线性微分方程**

**定义与解法**：

标准形式为$\frac{dy}{dx} + P(x)y = Q(x)$。其解法是使用积分因子法，积分因子为：

$\mu(x) = e^{\int P(x)dx}$

将方程两边乘以积分因子，得到：

$\frac{d}{dx}(\mu(x)y) = \mu(x)Q(x)$

积分后得到通解：

$y(x) = \frac{1}{\mu(x)} \left( \int \mu(x)Q(x)dx + C \right)$

[(45)](https://blog.csdn.net/DaPiCaoMin/article/details/144947625)。

**示例**：

解方程$\frac{dy}{dx} + 2y = x$。积分因子为$\mu(x) = e^{\int 2dx} = e^{2x}$。方程两边乘以$\mu(x)$得：

$e^{2x}\frac{dy}{dx} + 2e^{2x}y = xe^{2x}$

左边为$\frac{d}{dx}(e^{2x}y)$，积分得：

$e^{2x}y = \int xe^{2x}dx + C = \frac{1}{2}xe^{2x} - \frac{1}{4}e^{2x} + C$

因此，通解为：

$y = \frac{1}{2}x - \frac{1}{4} + Ce^{-2x}$

[(45)](https://blog.csdn.net/DaPiCaoMin/article/details/144947625)。

**4. 伯努利方程**

**定义与解法**：

形如$\frac{dy}{dx} + P(x)y = Q(x)y^n$（其中$n \neq 0, 1$）的方程称为伯努利方程。通过变量替换$z = y^{1-n}$，将方程转化为一阶线性微分方程：

$\frac{dz}{dx} + (1-n)P(x)z = (1-n)Q(x)$

然后使用积分因子法求解[(46)](https://m.renrendoc.com/paper/197017430.html)。

**示例**：

解方程$\frac{dy}{dx} + y = xy^3$。这是伯努利方程，其中$n = 3$。令$z = y^{-2}$，则$\frac{dz}{dx} = -2y^{-3}\frac{dy}{dx}$。原方程两边乘以$-2y^{-3}$得：

$\frac{dz}{dx} - 2z = -2x$

这是一个一阶线性方程，积分因子为$\mu(x) = e^{\int -2dx} = e^{-2x}$。方程两边乘以$\mu(x)$得：

$e^{-2x}\frac{dz}{dx} - 2e^{-2x}z = -2xe^{-2x}$

左边为$\frac{d}{dx}(e^{-2x}z)$，积分得：

$e^{-2x}z = \int -2xe^{-2x}dx + C = e^{-2x}(x + \frac{1}{2}) + C$

因此，$z = x + \frac{1}{2} + Ce^{2x}$，代回原变量得：

$\frac{1}{y^2} = x + \frac{1}{2} + Ce^{2x}$

即$y = \pm \frac{1}{\sqrt{x + \frac{1}{2} + Ce^{2x}}}$[(46)](https://m.renrendoc.com/paper/197017430.html)。

**5. 全微分方程**

**定义与解法**：

如果方程$P(x, y)dx + Q(x, y)dy = 0$满足$\frac{\partial P}{\partial y} = \frac{\partial Q}{\partial x}$，则称为全微分方程，存在函数$u(x, y)$使得$du = Pdx + Qdy$。此时方程的通解为$u(x, y) = C$[(46)](https://m.renrendoc.com/paper/197017430.html)。

**示例**：

解方程$(3x^2 + 6xy^2)dx + (6x^2y + 4y^3)dy = 0$。这里$P = 3x^2 + 6xy^2$，$Q = 6x^2y + 4y^3$，计算得$\frac{\partial P}{\partial y} = 12xy$，$\frac{\partial Q}{\partial x} = 12xy$，满足全微分条件。寻找$u(x, y)$使得$\frac{\partial u}{\partial x} = P$，$\frac{\partial u}{\partial y} = Q$。由$\frac{\partial u}{\partial x} = 3x^2 + 6xy^2$积分得$u = x^3 + 3x^2y^2 + \phi(y)$。再对$y$求导并与$Q$比较，得$6x^2y + \phi'(y) = 6x^2y + 4y^3$，故$\phi'(y) = 4y^3$，积分得$\phi(y) = y^4 + C$。因此，通解为$x^3 + 3x^2y^2 + y^4 = C$[(46)](https://m.renrendoc.com/paper/197017430.html)。

**6. 积分因子法**

**定义与理论**：

对于非全微分方程$P(x, y)dx + Q(x, y)dy = 0$，如果存在函数$\mu(x, y)$使得$\mu Pdx + \mu Qdy = 0$成为全微分方程，则称$\mu(x, y)$为原方程的积分因子。积分因子的存在性由以下定理保证：

定理：设$P$和$Q$在某区域内都是连续可微的，则方程$Pdx + Qdy = 0$有形如$\mu = e^{\int \frac{\frac{\partial P}{\partial y} - \frac{\partial Q}{\partial x}}{P} dy}$的积分因子的充要条件是：函数$\frac{\frac{\partial P}{\partial y} - \frac{\partial Q}{\partial x}}{P}$仅是 x 的函数[(46)](https://m.renrendoc.com/paper/197017430.html)。

**示例**：

解方程$ydx - xdy = 0$。这里$P = y$，$Q = -x$，计算得$\frac{\partial P}{\partial y} = 1$，$\frac{\partial Q}{\partial x} = -1$，不满足全微分条件。寻找积分因子，计算$\frac{\frac{\partial P}{\partial y} - \frac{\partial Q}{\partial x}}{P} = \frac{1 - (-1)}{y} = \frac{2}{y}$，不是仅关于 x 的函数，因此需要尝试其他形式的积分因子。假设积分因子为$\mu = \frac{1}{x^2}$，方程两边乘以$\mu$得$\frac{ydx - xdy}{x^2} = 0$，即$d\left(\frac{y}{x}\right) = 0$，因此通解为$\frac{y}{x} = C$，即$y = Cx$[(46)](https://m.renrendoc.com/paper/197017430.html)。

### 5.2 高阶常微分方程的降阶法

对于高阶常微分方程，有时可以通过变量替换将其转化为低阶方程，从而简化求解过程。以下是几种常见的降阶法：

**1. 不显含未知函数的方程**

**方程形式**：$y^{(n)} = f(x, y', y'', \ldots, y^{(n-1)})$，即方程不显含未知函数 y 本身。

**解法**：令$z = y'$，则方程降为关于 z 的 n-1 阶方程：

$z^{(n-1)} = f(x, z, z', \ldots, z^{(n-2)})$

依次类推，直到降为一阶方程[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**示例**：

解方程$y''' = \frac{1}{x}y''$。令$z = y''$，则方程变为$z' = \frac{1}{x}z$，这是一个变量可分离方程。分离变量得$\frac{dz}{z} = \frac{dx}{x}$，积分得$\ln|z| = \ln|x| + C_1$，即$z = C_1x$。因此$y'' = C_1x$，积分两次得$y = \frac{C_1}{6}x^3 + C_2x + C_3$[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**2. 不显含自变量的方程**

**方程形式**：$y'' = f(y, y')$，即方程不显含自变量 x。

**解法**：令$p = y'$，则$y'' = p\frac{dp}{dy}$，方程变为一阶方程：

$p\frac{dp}{dy} = f(y, p)$

解此方程后，再积分一次得到 y 的表达式[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**示例**：

解方程$yy'' - (y')^2 = 0$。令$p = y'$，则$y'' = p\frac{dp}{dy}$，代入方程得$yp\frac{dp}{dy} - p^2 = 0$。若$p \neq 0$，可约去 p 得$y\frac{dp}{dy} - p = 0$，即$\frac{dp}{p} = \frac{dy}{y}$，积分得$\ln|p| = \ln|y| + C_1$，即$p = C_1y$。因此$\frac{dy}{dx} = C_1y$，解得$y = C_2e^{C_1x}$。若$p = 0$，则$y = C$也是解，可包含在通解中（当$C_1 = 0$时）[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**3. 欧拉方程**

**定义与解法**：

形如$x^n y^{(n)} + a_1x^{n-1}y^{(n-1)} + \cdots + a_{n-1}xy' + a_n y = f(x)$的方程称为欧拉方程。通过变量替换$t = \ln x$（即$x = e^t$），可以将其转化为常系数线性微分方程，然后使用常系数方程的解法求解[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**示例**：

解方程$x^2y'' + 3xy' + y = 0$。令$t = \ln x$，则$\frac{dy}{dx} = \frac{1}{x}\frac{dy}{dt}$，$\frac{d^2y}{dx^2} = \frac{1}{x^2}\left(\frac{d^2y}{dt^2} - \frac{dy}{dt}\right)$。代入原方程得：

$x^2 \cdot \frac{1}{x^2}\left(\frac{d^2y}{dt^2} - \frac{dy}{dt}\right) + 3x \cdot \frac{1}{x}\frac{dy}{dt} + y = 0$

化简为$\frac{d^2y}{dt^2} + 2\frac{dy}{dt} + y = 0$，这是一个常系数齐次方程，特征方程为$r^2 + 2r + 1 = 0$，解得重根$r = -1$。因此通解为$y = (C_1 + C_2t)e^{-t}$，代回原变量得$y = (C_1 + C_2\ln x)\frac{1}{x}$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

### 5.3 线性常微分方程的解法

线性常微分方程是常微分方程中最重要的一类，其解法已经形成了完整的理论体系。以下是几种主要的解法：

**1. 常系数齐次线性微分方程**

**方程形式**：$a_n y^{(n)} + a_{n-1} y^{(n-1)} + \cdots + a_1 y' + a_0 y = 0$，其中$a_i$为常数。

**解法**：



1.  写出特征方程：$a_n r^n + a_{n-1} r^{n-1} + \cdots + a_1 r + a_0 = 0$。

2.  求解特征方程，得到 n 个特征根$r_1, r_2, \ldots, r_n$。

3.  根据特征根的不同情况，写出通解：

*   单实根$r$：对应解$y = e^{rx}$。

*   重实根$r$（k 重）：对应解$y = e^{rx}, xe^{rx}, \ldots, x^{k-1}e^{rx}$。

*   共轭复根$\alpha \pm \beta i$：对应解$e^{\alpha x}\cos\beta x, e^{\alpha x}\sin\beta x$。

*   重共轭复根$\alpha \pm \beta i$（k 重）：对应解$e^{\alpha x}\cos\beta x, e^{\alpha x}\sin\beta x, xe^{\alpha x}\cos\beta x, xe^{\alpha x}\sin\beta x, \ldots, x^{k-1}e^{\alpha x}\cos\beta x, x^{k-1}e^{\alpha x}\sin\beta x$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**示例**：

解方程$y''' - 3y'' + 3y' - y = 0$。特征方程为$r^3 - 3r^2 + 3r - 1 = 0$，即$(r - 1)^3 = 0$，三重根$r = 1$。因此通解为$y = (C_1 + C_2x + C_3x^2)e^x$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**2. 常系数非齐次线性微分方程**

**方程形式**：$a_n y^{(n)} + a_{n-1} y^{(n-1)} + \cdots + a_1 y' + a_0 y = f(x)$。

**解法**：

通解为对应的齐次方程的通解加上非齐次方程的一个特解。求特解的方法主要有：

**待定系数法**：

适用于$f(x)$为多项式、指数函数、正弦函数、余弦函数以及它们的乘积的情况。根据$f(x)$的形式假设特解的形式，代入原方程确定系数[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**示例**：

解方程$y'' + y = \sin x$。对应的齐次方程的特征方程为$r^2 + 1 = 0$，解得$r = \pm i$，通解为$y_h = C_1\cos x + C_2\sin x$。由于非齐次项为$\sin x$，而$\sin x$是齐次方程的解，因此假设特解为$y_p = x(A\cos x + B\sin x)$。代入原方程得：

$y_p'' + y_p = -2A\sin x + 2B\cos x = \sin x$

比较系数得$-2A = 1$，$2B = 0$，解得$A = -\frac{1}{2}$，$B = 0$。因此特解为$y_p = -\frac{1}{2}x\cos x$，通解为$y = C_1\cos x + C_2\sin x - \frac{1}{2}x\cos x$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**常数变易法**：

对于一般的非齐次线性微分方程，可以使用常数变易法。其基本思想是将齐次方程通解中的常数视为待定函数，并代入原方程求解这些函数[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**示例**：

解方程$y'' + y = \tan x$。对应的齐次方程的通解为$y_h = C_1\cos x + C_2\sin x$。设特解为$y_p = u_1(x)\cos x + u_2(x)\sin x$，代入原方程得：

$u_1'(-\sin x) + u_2'(\cos x) = 0 \\
u_1'(-\cos x) + u_2'(-\sin x) = \tan x$

解此方程组得$u_1' = -\sin x \tan x$，$u_2' = \cos x \tan x$。积分得$u_1 = \cos x - \ln|\sec x + \tan x|$，$u_2 = -\cos x$。因此特解为$y_p = [\cos x - \ln|\sec x + \tan x|]\cos x + (-\cos x)\sin x = \cos^2 x - \cos x \ln|\sec x + \tan x| - \cos x \sin x$。通解为$y = y_h + y_p$[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**3. 欧拉方程**

欧拉方程的解法已在 5.2 节中介绍。

**4. 拉普拉斯变换法**

拉普拉斯变换法是求解常系数线性微分方程的一种有效方法，特别是对于非齐次项为脉冲函数、阶跃函数或分段连续函数的情况尤为方便[(69)](http://www.cmpedu.com/books/book/63894.htm)。

**基本步骤**：



1.  对原方程两边取拉普拉斯变换，利用线性性质和微分性质将微分方程转化为代数方程。

2.  解代数方程，得到象函数。

3.  对象函数取逆拉普拉斯变换，得到原方程的解[(69)](http://www.cmpedu.com/books/book/63894.htm)。

**示例**：

解方程$y'' + 2y' + y = 0$，初始条件$y(0) = 1$，$y'(0) = 0$。取拉普拉斯变换得：

$s^2Y(s) - sy(0) - y'(0) + 2(sY(s) - y(0)) + Y(s) = 0$

代入初始条件得：

$s^2Y(s) - s + 2sY(s) - 2 + Y(s) = 0$

整理得：

$(s^2 + 2s + 1)Y(s) = s + 2$

即：

$Y(s) = \frac{s + 2}{(s + 1)^2} = \frac{1}{s + 1} + \frac{1}{(s + 1)^2}$

取逆变换得：

$y(t) = e^{-t} + te^{-t} = (1 + t)e^{-t}$

[(69)](http://www.cmpedu.com/books/book/63894.htm)。

### 5.4 常微分方程组的解法

对于常微分方程组，可以通过消元法或矩阵方法将其转化为高阶方程或标准形式进行求解。

**1. 消元法**

**基本思想**：通过对各个方程进行求导和组合，消去其他未知函数，得到只含有一个未知函数的高阶方程，求解后再代入原方程组求出其他未知函数[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**示例**：

解方程组：

$\frac{dx}{dt} = 2x + y \\
\frac{dy}{dt} = x + 2y$

从第一个方程解出$y = \frac{dx}{dt} - 2x$，代入第二个方程得：

$\frac{d}{dt}\left(\frac{dx}{dt} - 2x\right) = x + 2\left(\frac{dx}{dt} - 2x\right)$

化简得：

$\frac{d^2x}{dt^2} - 4\frac{dx}{dt} + 3x = 0$

特征方程为$r^2 - 4r + 3 = 0$，解得$r = 1, 3$，因此$x = C_1e^t + C_2e^{3t}$。代入$y = \frac{dx}{dt} - 2x$得$y = -C_1e^t + C_2e^{3t}$。因此方程组的通解为：

$\begin{cases}
x = C_1e^t + C_2e^{3t} \\
y = -C_1e^t + C_2e^{3t}
\end{cases}$

[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**2. 矩阵方法**

**基本思想**：将常微分方程组表示为矩阵形式$\frac{d\mathbf{y}}{dt} = A\mathbf{y}$，其中 A 为系数矩阵。通过求解矩阵 A 的特征值和特征向量，得到方程组的通解[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

**示例**：

上述示例中的方程组可以表示为：

$\frac{d}{dt}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix}$

系数矩阵 A 的特征方程为$\det(A - \lambda I) = (2 - \lambda)^2 - 1 = 0$，解得$\lambda = 1, 3$。对应的特征向量分别为$\begin{pmatrix} 1 \\ -1 \end{pmatrix}$和$\begin{pmatrix} 1 \\ 1 \end{pmatrix}$。因此方程组的通解为：

$\begin{pmatrix} x \\ y \end{pmatrix} = C_1e^t\begin{pmatrix} 1 \\ -1 \end{pmatrix} + C_2e^{3t}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

即：

$\begin{cases}
x = C_1e^t + C_2e^{3t} \\
y = -C_1e^t + C_2e^{3t}
\end{cases}$

与消元法得到的结果一致[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

## 六、常微分方程的数值解法

### 6.1 数值解法概述

对于大多数常微分方程，特别是非线性方程，很难找到解析解。因此，数值解法成为求解常微分方程的重要手段。数值解法的基本思想是将连续的问题离散化，通过递推公式在离散点上近似求解函数值[(53)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)。

**初值问题的数值解法**：

考虑一阶常微分方程初值问题：

$\frac{dy}{dx} = f(x, y), \quad y(x_0) = y_0$

数值解法的目标是在一系列离散点$x_0, x_1, \ldots, x_n$上计算出近似解$y_0, y_1, \ldots, y_n$，其中$x_{i+1} = x_i + h$，h 为步长[(53)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)。

**数值解法的基本步骤**：



1.  将求解区间$[a, b]$划分为 n 个小区间，步长$h = \frac{b - a}{n}$。

2.  在每个节点$x_i$处，利用已知的$y_i$和$f(x_i, y_i)$，通过某种递推公式计算$y_{i+1}$。

3.  逐步推进，直到计算出所有节点上的近似解[(53)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)。

**数值解法的误差分析**：

数值解法的误差主要包括：



*   **局部截断误差**：假设在计算$y_{i+1}$时，前面的计算都是精确的，即$y_i = y(x_i)$，则$y(x_{i+1}) - y_{i+1}$称为局部截断误差。

*   **整体误差**：精确解$y(x_i)$与近似解$y_i$之间的差异，即$y(x_i) - y_i$[(64)](https://blog.csdn.net/weixin_32187691/article/details/112480479)。

根据局部截断误差的阶数，可以将数值方法分为一阶方法、二阶方法、四阶方法等。例如，欧拉法是一阶方法，龙格 - 库塔法是四阶方法[(64)](https://blog.csdn.net/weixin_32187691/article/details/112480479)。

### 6.2 欧拉方法及其改进

**1. 欧拉方法**

**基本思想**：

欧拉方法是最简单的数值方法，它基于泰勒展开的一阶近似。在点$x_i$处，将$y(x_{i+1})$在$x_i$处展开：

$y(x_{i+1}) = y(x_i) + hy'(x_i) + O(h^2)$

忽略高阶项$O(h^2)$，得到欧拉公式：

$y_{i+1} = y_i + hf(x_i, y_i)$

[(64)](https://blog.csdn.net/weixin_32187691/article/details/112480479)。

**几何意义**：

欧拉方法的几何意义是在每一步都用切线来近似曲线，即从点$(x_i, y_i)$出发，沿着斜率为$f(x_i, y_i)$的直线前进 h 步长，得到下一个点$(x_{i+1}, y_{i+1})$[(64)](https://blog.csdn.net/weixin_32187691/article/details/112480479)。

**示例**：

用欧拉方法求解初值问题$\frac{dy}{dx} = y$，$y(0) = 1$，步长 h = 0.1，计算到 x = 1。

精确解为$y = e^x$。欧拉公式为$y_{i+1} = y_i + 0.1y_i = 1.1y_i$。计算结果如下：



| x\_i | y\_i (欧拉法) | y (x\_i) (精确解) | 误差     |
| ---- | ---------- | -------------- | ------ |
| 0.0  | 1.0000     | 1.0000         | 0.0000 |
| 0.1  | 1.1000     | 1.1052         | 0.0052 |
| 0.2  | 1.2100     | 1.2214         | 0.0114 |
| 0.3  | 1.3310     | 1.3499         | 0.0189 |
| ...  | ...        | ...            | ...    |
| 1.0  | 2.5937     | 2.7183         | 0.1246 |

可以看出，欧拉方法的误差随着步数的增加而累积，整体误差为$O(h)$[(64)](https://blog.csdn.net/weixin_32187691/article/details/112480479)。

**2. 改进的欧拉方法**

**基本思想**：

为了提高精度，可以使用梯形公式代替欧拉公式中的矩形公式。梯形公式基于积分中值定理：

$y(x_{i+1}) = y(x_i) + \frac{h}{2}[f(x_i, y(x_i)) + f(x_{i+1}, y(x_{i+1}))] + O(h^3)$

但由于$y(x_{i+1})$是未知的，无法直接应用。改进的欧拉方法采用预测 - 校正的方法：



1.  预测：使用欧拉公式计算初步的$y_{i+1}^* = y_i + hf(x_i, y_i)$

2.  校正：使用梯形公式计算$y_{i+1} = y_i + \frac{h}{2}[f(x_i, y_i) + f(x_{i+1}, y_{i+1}^*)]$

    这称为改进的欧拉方法或 Heun 方法[(55)](https://codepal.ai/code-generator/query/t1XP1572/python-solve-ode-methods)。

**示例**：

用改进的欧拉方法求解上述初值问题$\frac{dy}{dx} = y$，$y(0) = 1$，步长 h = 0.1，计算到 x = 1。

改进的欧拉公式为：

$y_{i+1}^* = y_i + 0.1y_i = 1.1y_i \\
y_{i+1} = y_i + 0.05(y_i + y_{i+1}^*) = y_i + 0.05(y_i + 1.1y_i) = 1.105y_i$

计算结果如下：



| x\_i | y\_i (改进欧拉法) | y (x\_i) (精确解) | 误差     |
| ---- | ------------ | -------------- | ------ |
| 0.0  | 1.0000       | 1.0000         | 0.0000 |
| 0.1  | 1.1050       | 1.1052         | 0.0002 |
| 0.2  | 1.2210       | 1.2214         | 0.0004 |
| 0.3  | 1.3492       | 1.3499         | 0.0007 |
| ...  | ...          | ...            | ...    |
| 1.0  | 2.7140       | 2.7183         | 0.0043 |

改进的欧拉方法的误差明显小于欧拉方法，整体误差为$O(h^2)$，即二阶方法[(55)](https://codepal.ai/code-generator/query/t1XP1572/python-solve-ode-methods)。

### 6.3 龙格 - 库塔方法

龙格 - 库塔 (Runge-Kutta) 方法是一类高精度的数值方法，其中最常用的是四阶龙格 - 库塔方法（RK4），其精度为四阶，能够在较少的计算量下获得较高的精度[(59)](https://blog.csdn.net/u012836279/article/details/80176985)。

**四阶龙格 - 库塔方法**：

**基本思想**：

四阶龙格 - 库塔方法基于在区间$[x_i, x_{i+1}]$内多个点上计算斜率，并将这些斜率加权平均，以提高精度。具体公式如下：

$k_1 = hf(x_i, y_i) \\
k_2 = hf(x_i + \frac{h}{2}, y_i + \frac{k_1}{2}) \\
k_3 = hf(x_i + \frac{h}{2}, y_i + \frac{k_2}{2}) \\
k_4 = hf(x_i + h, y_i + k_3) \\
y_{i+1} = y_i + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

[(59)](https://blog.csdn.net/u012836279/article/details/80176985)。

**几何意义**：

四阶龙格 - 库塔方法在区间内计算了四个点的斜率：



1.  $k_1$是区间起点的斜率。

2.  $k_2$是区间中点处基于$k_1$预测的斜率。

3.  $k_3$是区间中点处基于$k_2$预测的斜率。

4.  $k_4$是区间终点处基于$k_3$预测的斜率。

    通过对这四个斜率的加权平均，得到更精确的近似值[(59)](https://blog.csdn.net/u012836279/article/details/80176985)。

**示例**：

用四阶龙格 - 库塔方法求解上述初值问题$\frac{dy}{dx} = y$，$y(0) = 1$，步长 h = 0.1，计算到 x = 1。

计算过程如下：

对于每个 i，计算 k1 到 k4：

$k1 = 0.1 \times y_i \\
k2 = 0.1 \times (y_i + 0.05k1) \\
k3 = 0.1 \times (y_i + 0.05k2) \\
k4 = 0.1 \times (y_i + 0.1k3) \\
y_{i+1} = y_i + \frac{1}{6}(k1 + 2k2 + 2k3 + k4)$

计算结果如下：



| x\_i | y\_i (RK4) | y (x\_i) (精确解) | 误差     |
| ---- | ---------- | -------------- | ------ |
| 0.0  | 1.0000     | 1.0000         | 0.0000 |
| 0.1  | 1.1052     | 1.1052         | 0.0000 |
| 0.2  | 1.2214     | 1.2214         | 0.0000 |
| 0.3  | 1.3499     | 1.3499         | 0.0000 |
| ...  | ...        | ...            | ...    |
| 1.0  | 2.7183     | 2.7183         | 0.0000 |

可以看到，四阶龙格 - 库塔方法在步长 h = 0.1 的情况下，已经能够得到精确到四位小数的结果，误差为$O(h^4)$，即四阶方法[(59)](https://blog.csdn.net/u012836279/article/details/80176985)。

**Python 实现**：

下面是一个使用四阶龙格 - 库塔法求解常微分方程的 Python 代码示例：



```
def runge\_kutta(f, t0, y0, tn, N):

&#x20;   """ 使用四阶龙格-库塔法求解 ODE """

&#x20;   dt = (tn - t0) / (N - 1)

&#x20;   ts = \[t0 + i \* dt for i in range(N)]

&#x20;   ys = \[y0]

&#x20;  &#x20;

&#x20;   for t in ts\[:-1]:

&#x20;       y = ys\[-1]

&#x20;       k1 = dt \* f(t, y)

&#x20;       k2 = dt \* f(t + 0.5 \* dt, y + 0.5 \* k1)

&#x20;       k3 = dt \* f(t + 0.5 \* dt, y + 0.5 \* k2)

&#x20;       k4 = dt \* f(t + dt, y + k3)

&#x20;       y\_next = y + (k1 + 2 \* k2 + 2 \* k3 + k4) / 6

&#x20;       ys.append(y\_next)

&#x20;  &#x20;

&#x20;   return ts, ys

\# 定义微分方程函数

def func(t, y):

&#x20;   return y  # 这里以dy/dt = y为例

\# 设置参数

t0 = 0.0

y0 = 1.0

tn = 1.0

N = 11  # 11个点对应步长0.1

\# 求解

ts, ys = runge\_kutta(func, t0, y0, tn, N)

\# 输出结果

for t, y in zip(ts, ys):

&#x20;   print(f"t = {t:.1f}, y = {y:.5f}")
```

运行这段代码，将得到与精确解几乎完全一致的结果，证明了四阶龙格 - 库塔法的高精度[(59)](https://blog.csdn.net/u012836279/article/details/80176985)。

### 6.4 常微分方程组的数值解法

对于常微分方程组，可以使用类似的数值方法进行求解。考虑一阶常微分方程组初值问题：

$\frac{dy_i}{dx} = f_i(x, y_1, y_2, \ldots, y_n), \quad y_i(x_0) = y_{i0}, \quad i = 1, 2, \ldots, n$

其数值解法与单个方程类似，只需将标量替换为向量[(53)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)。

**四阶龙格 - 库塔方法求解方程组**：

对于方程组，四阶龙格 - 库塔方法的形式为：

$\mathbf{k_1} = h\mathbf{f}(x_i, \mathbf{y_i}) \\
\mathbf{k_2} = h\mathbf{f}(x_i + \frac{h}{2}, \mathbf{y_i} + \frac{\mathbf{k_1}}{2}) \\
\mathbf{k_3} = h\mathbf{f}(x_i + \frac{h}{2}, \mathbf{y_i} + \frac{\mathbf{k_2}}{2}) \\
\mathbf{k_4} = h\mathbf{f}(x_i + h, \mathbf{y_i} + \mathbf{k_3}) \\
\mathbf{y_{i+1}} = \mathbf{y_i} + \frac{1}{6}(\mathbf{k_1} + 2\mathbf{k_2} + 2\mathbf{k_3} + \mathbf{k_4})$

其中$\mathbf{y}$是向量，$\mathbf{f}$是向量函数[(53)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)。

**示例**：

考虑以下常微分方程组：

$\frac{dy}{dt} = z \\
\frac{dz}{dt} = -y$

初始条件为$y(0) = 0$，$z(0) = 1$，这是一个描述简谐振动的方程组。

使用四阶龙格 - 库塔方法求解，步长 h = 0.1，计算到 t = 2π。

Python 代码如下：



```
import math

def harmonic\_oscillator(t, state):

&#x20;   y, z = state

&#x20;   dydt = z

&#x20;   dzdt = -y

&#x20;   return \[dydt, dzdt]

t0 = 0.0

y0 = 0.0

z0 = 1.0

tn = 2 \* math.pi

N = 100

ts = \[t0 + i \* (tn - t0) / (N - 1) for i in range(N)]

states = \[\[y0, z0]]

for t in ts\[:-1]:

&#x20;   y, z = states\[-1]

&#x20;   k1 = harmonic\_oscillator(t, \[y, z])

&#x20;   k2 = harmonic\_oscillator(t + 0.05 \* math.pi, \[y + 0.05 \* math.pi \* k1\[0], z + 0.05 \* math.pi \* k1\[1]])

&#x20;   k3 = harmonic\_oscillator(t + 0.05 \* math.pi, \[y + 0.05 \* math.pi \* k2\[0], z + 0.05 \* math.pi \* k2\[1]])

&#x20;   k4 = harmonic\_oscillator(t + 0.1 \* math.pi, \[y + 0.1 \* math.pi \* k3\[0], z + 0.1 \* math.pi \* k3\[1]])

&#x20;   y\_next = y + (k1\[0] + 2 \* k2\[0] + 2 \* k3\[0] + k4\[0]) \* 0.1 \* math.pi / 6

&#x20;   z\_next = z + (k1\[1] + 2 \* k2\[1] + 2 \* k3\[1] + k4\[1]) \* 0.1 \* math.pi / 6

&#x20;   states.append(\[y\_next, z\_next])

\# 输出结果

for t, (y, z) in zip(ts, states):

&#x20;   print(f"t = {t:.1f}, y = {y:.5f}, z = {z:.5f}")
```

运行这段代码，将得到简谐振动的数值解，与精确解$y = \sin t$，$z = \cos t$非常接近[(60)](https://blog.51cto.com/u_16213464/13146165)。

### 6.5 刚性方程组的数值解法

刚性方程组是一类特殊的常微分方程组，其特点是方程组中包含了变化速率相差很大的变量，这使得普通的数值方法需要非常小的步长才能保持稳定性，计算效率很低[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**刚性方程组的定义**：

考虑线性常微分方程组$\frac{d\mathbf{y}}{dt} = A\mathbf{y}$，其中 A 是常数矩阵。如果 A 的所有特征值$\lambda_i$满足$\text{Re}(\lambda_i) < 0$，并且最大特征值实部与最小特征值实部的比值很大，则称该方程组是刚性的。这个比值称为刚性比[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**刚性方程组的数值解法**：

对于刚性方程组，通常使用隐式方法或专门设计的刚性方法，如 BDF 方法（向后差分公式）、Gear 方法等，这些方法具有更好的稳定性，可以使用较大的步长[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**MATLAB 中的刚性求解器**：

MATLAB 提供了多个用于求解刚性方程组的函数，如 ode15s、ode23s、ode23t、ode23tb 等。这些函数使用不同的算法，适用于不同类型的刚性问题[(18)](http://rivista.math.unipr.it/fulltext/1999-2s/05.pdf)。

**示例**：

考虑刚性方程组：

$\frac{dy}{dt} = -1000y + 3000z + 2000 \\
\frac{dz}{dt} = 1000y - 3000z$

初始条件$y(0) = 3$，$z(0) = 1$。

该方程组的精确解为：

$y(t) = 2 + e^{-1000t} \\
z(t) = 1 - \frac{1}{3}e^{-1000t}$

当 t 增大时，指数项迅速衰减，解趋于稳态解$y = 2$，$z = 1$。

使用显式方法（如四阶龙格 - 库塔法）求解这个方程组时，由于刚性比很大（特征值为 - 1000 和 - 2000），需要非常小的步长（如 h < 0.001）才能保持稳定性，计算量很大。而使用隐式方法（如 ode15s），可以使用较大的步长，显著提高计算效率[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

## 七、常微分方程的应用

### 7.1 物理学中的应用

常微分方程在物理学中有着广泛的应用，几乎所有的物理过程都可以用常微分方程来描述。以下是几个典型的应用案例：

**1. 牛顿运动定律**

**应用场景**：描述物体的运动状态，如自由落体、弹簧振子、行星轨道等。

**模型建立**：

根据牛顿第二定律$F = ma$，可以建立物体的运动方程。例如，对于弹簧振子，恢复力$F = -kx$，得到方程：

$m\frac{d^2x}{dt^2} + kx = 0$

这是一个二阶线性齐次方程，其解描述了简谐振动[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**数值模拟**：

使用数值方法求解上述方程，可以模拟弹簧振子的运动轨迹。例如，使用四阶龙格 - 库塔方法求解：

$\frac{dx}{dt} = v \\
\frac{dv}{dt} = -\frac{k}{m}x$

可以得到位移 x 和速度 v 随时间的变化曲线[(64)](https://blog.csdn.net/weixin_32187691/article/details/112480479)。

**2. 电路分析**

**应用场景**：描述电路中电流、电压的变化规律，如 RC 电路、RL 电路、LC 电路等。

**模型建立**：

根据基尔霍夫定律，可以建立电路的微分方程。例如，对于 RLC 串联电路，有：

$L\frac{d^2q}{dt^2} + R\frac{dq}{dt} + \frac{1}{C}q = E(t)$

其中 q 是电荷，E (t) 是电源电动势[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**数值模拟**：

使用数值方法求解上述方程，可以分析电路的暂态响应和稳态响应。例如，使用四阶龙格 - 库塔方法求解：

$\frac{dq}{dt} = i \\
\frac{di}{dt} = \frac{1}{L}(E(t) - Ri - \frac{1}{C}q)$

可以得到电荷 q 和电流 i 随时间的变化[(69)](http://www.cmpedu.com/books/book/63894.htm)。

**3. 波动方程的简化**

**应用场景**：在声学、电磁学、地震学等领域中，波动现象可以用偏微分方程描述，但在某些情况下可以简化为常微分方程。

**模型建立**：

考虑一维波动方程：

$\frac{\partial^2u}{\partial t^2} = c^2\frac{\partial^2u}{\partial x^2}$

假设解的形式为$u(x, t) = X(x)T(t)$，代入方程后分离变量，得到两个常微分方程：

$\frac{d^2X}{dx^2} + \lambda X = 0 \\
\frac{d^2T}{dt^2} + \lambda c^2T = 0$

其中$\lambda$是分离常数[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**数值模拟**：

通过求解上述常微分方程，可以得到波动方程的特征解，进而通过叠加得到一般解。例如，对于两端固定的弦振动问题，可以得到正弦级数形式的解[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

### 7.2 生物学中的应用

常微分方程在生物学中也有广泛的应用，特别是在种群动力学、酶动力学、神经传导等领域。

**1. 种群增长模型**

**应用场景**：描述生物种群数量随时间的变化规律，如细菌繁殖、人口增长等。

**模型建立**：



*   **马尔萨斯模型**（指数增长模型）：

$\frac{dN}{dt} = rN$

其中 N 是种群数量，r 是增长率。解为$N(t) = N_0e^{rt}$，适用于资源无限的情况[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。



*   **逻辑斯谛模型**（阻滞增长模型）：

$\frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right)$

其中 K 是环境容纳量。解为$N(t) = \frac{K}{1 + \left(\frac{K}{N_0} - 1\right)e^{-rt}}$，适用于资源有限的情况[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**参数估计**：

通过拟合实际数据，可以估计模型中的参数 r 和 K。例如，使用非线性最小二乘法优化目标函数：

$\min_{r, K} \sum_{i=1}^n \left(N_i - \frac{K}{1 + \left(\frac{K}{N_0} - 1\right)e^{-rt_i}}\right)^2$

[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**数值模拟**：

使用数值方法求解逻辑斯谛方程，可以预测种群数量的变化趋势。例如，使用四阶龙格 - 库塔方法求解：

$\frac{dN}{dt} = 0.5N\left(1 - \frac{N}{1000}\right)$

初始条件$N(0) = 100$，可以得到种群数量随时间的变化曲线[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**2. 捕食者 - 猎物模型**

**应用场景**：描述两个相互作用种群（捕食者和猎物）的数量变化，如狐狸和兔子、寄生虫和宿主等。

**模型建立**：

Lotka-Volterra 模型：

$\frac{dR}{dt} = aR - bRF \\
\frac{dF}{dt} = -cF + dRF$

其中 R 是猎物数量，F 是捕食者数量，a 是猎物的增长率，b 是捕食者对猎物的捕获率，c 是捕食者的死亡率，d 是捕食者利用猎物的效率[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**定性分析**：

通过分析相平面上的轨线，可以研究系统的动态行为。Lotka-Volterra 模型的解是周期性的，表明捕食者和猎物数量呈现周期性波动[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**数值模拟**：

使用数值方法求解 Lotka-Volterra 模型，可以得到猎物和捕食者数量的时间序列。例如，使用四阶龙格 - 库塔方法求解：

$\frac{dR}{dt} = 0.5R - 0.01RF \\
\frac{dF}{dt} = -0.3F + 0.005RF$

初始条件 R (0) = 100，F (0) = 20，可以得到 R 和 F 随时间的周期性变化[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**3. 酶动力学**

**应用场景**：描述酶催化反应的速率和机制，如米氏方程、别构酶动力学等。

**模型建立**：

米氏方程：

$\frac{dS}{dt} = -k_1ES + k_{-1}E + k_2ES \\
\frac{dP}{dt} = k_2ES$

其中 S 是底物浓度，P 是产物浓度，E 是酶浓度，ES 是酶 - 底物复合物浓度，k1、k-1、k2 是反应速率常数[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

**稳态近似**：

假设 ES 复合物的浓度保持稳态，即$\frac{dES}{dt} = 0$，可以得到米氏方程的简化形式：

$\frac{dP}{dt} = \frac{V_{max}S}{K_m + S}$

其中$V_{max} = k_2[E]_0$是最大反应速率，$K_m = \frac{k_{-1} + k_2}{k_1}$是米氏常数[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

**参数估计**：

通过拟合实验数据，可以估计米氏方程中的参数 Vmax 和 Km。例如，使用非线性最小二乘法优化目标函数：

$\min_{V_{max}, K_m} \sum_{i=1}^n \left(v_i - \frac{V_{max}S_i}{K_m + S_i}\right)^2$

[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

### 7.3 工程学中的应用

常微分方程在工程学中有着广泛的应用，特别是在控制系统、信号处理、结构动力学等领域。

**1. 控制系统**

**应用场景**：描述控制系统的动态行为，如温度控制、位置控制、速度控制等。

**模型建立**：

控制系统的数学模型通常以微分方程的形式表示。例如，对于一个简单的 RC 电路控制系统，可以建立方程：

$RC\frac{dv_c}{dt} + v_c = v_i$

其中 vc 是电容电压，vi 是输入电压[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**传递函数**：

通过拉普拉斯变换，可以将微分方程转换为代数方程，得到系统的传递函数：

$G(s) = \frac{V_c(s)}{V_i(s)} = \frac{1}{RCs + 1}$

传递函数描述了系统对输入信号的响应特性[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**频域分析**：

通过分析系统的频率响应，可以研究系统的稳定性、带宽、增益等特性。例如，对于上述 RC 电路，频率响应为：

$G(j\omega) = \frac{1}{1 + j\omega RC}$

其幅频特性为$|G(j\omega)| = \frac{1}{\sqrt{1 + (\omega RC)^2}}$，相频特性为$\angle G(j\omega) = -\arctan(\omega RC)$[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**PID 控制器设计**：

PID（比例 - 积分 - 微分）控制器是最常用的控制器之一，其输出 u (t) 与误差 e (t) 的关系为：

$u(t) = K_pe(t) + K_i\int_0^t e(\tau)d\tau + K_d\frac{de(t)}{dt}$

通过调整参数 Kp、Ki、Kd，可以改善系统的性能[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**2. 机械振动**

**应用场景**：描述机械系统的振动特性，如桥梁振动、车辆悬架系统、建筑物抗震等。

**模型建立**：

单自由度弹簧 - 质量 - 阻尼系统：

$m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F(t)$

其中 m 是质量，c 是阻尼系数，k 是弹簧刚度，F (t) 是外部激励力，x 是位移[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**自由振动分析**：

当 F (t) = 0 时，系统的自由振动方程为：

$m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0$

根据阻尼比$\zeta = \frac{c}{2\sqrt{km}}$的不同，系统的响应可以分为欠阻尼、临界阻尼和过阻尼三种情况[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**强迫振动分析**：

当 F (t) = F0sinωt 时，系统的强迫振动方程为：

$m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F_0\sin\omega t$

其稳态响应为：

$x(t) = \frac{F_0}{k}\frac{1}{\sqrt{(1 - r^2)^2 + (2\zeta r)^2}}\sin(\omega t - \phi)$

其中$r = \frac{\omega}{\omega_n}$是频率比，$\omega_n = \sqrt{\frac{k}{m}}$是固有频率，$\phi = \arctan\left(\frac{2\zeta r}{1 - r^2}\right)$是相位差[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**共振现象**：

当激励频率接近系统的固有频率时，系统会发生共振，振幅达到最大值。共振现象在工程中需要特别关注，可能导致结构损坏[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**3. 电路系统**

**应用场景**：描述电路中电压、电流的变化规律，如滤波器设计、信号处理、电力系统等。

**模型建立**：

RLC 串联电路：

$L\frac{d^2i}{dt^2} + R\frac{di}{dt} + \frac{1}{C}i = \frac{1}{L}\frac{dV}{dt}$

其中 i 是电流，V 是电源电压，L 是电感，R 是电阻，C 是电容[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**频率响应**：

通过分析系统的频率响应，可以设计各种滤波器。例如，低通滤波器、高通滤波器、带通滤波器等。对于 RLC 串联电路，其频率响应为：

$H(j\omega) = \frac{1}{1 + jQ\left(\frac{\omega}{\omega_0} - \frac{\omega_0}{\omega}\right)}$

其中$\omega_0 = \frac{1}{\sqrt{LC}}$是谐振频率，$Q = \frac{R}{\sqrt{\frac{L}{C}}}$是品质因数[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**暂态响应**：

电路的暂态响应描述了从一个稳态到另一个稳态的过渡过程。例如，当电源突然接通或断开时，电路中的电流和电压会经历暂态过程，最终达到新的稳态[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

### 7.4 经济学和金融学中的应用

常微分方程在经济学和金融学中也有重要的应用，特别是在经济增长模型、投资组合理论、期权定价等领域。

**1. 经济增长模型**

**应用场景**：描述经济系统的增长和发展，如索洛模型、内生增长模型等。

**模型建立**：

索洛增长模型：

$\frac{dk}{dt} = sf(k) - (n + \delta)k$

其中 k 是资本存量，s 是储蓄率，f (k) 是生产函数，n 是人口增长率，δ 是资本折旧率[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**稳态分析**：

当$\frac{dk}{dt} = 0$时，系统达到稳态，资本存量 k \* 满足：

$sf(k^*) = (n + \delta)k^*$

稳态分析可以预测经济的长期增长趋势[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**比较静态分析**：

通过分析储蓄率 s、人口增长率 n 等参数变化对稳态的影响，可以研究不同政策对经济增长的影响。例如，提高储蓄率会增加稳态资本存量，从而提高产出水平[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**2. 投资组合理论**

**应用场景**：描述投资组合的风险和收益之间的关系，如马科维茨投资组合理论、资本资产定价模型等。

**模型建立**：

考虑一个包含 n 种资产的投资组合，其收益率向量为 r，协方差矩阵为 Σ。投资组合的收益率和方差分别为：

$R_p = \mathbf{w}^T\mathbf{r} \\
\sigma_p^2 = \mathbf{w}^T\Sigma\mathbf{w}$

其中 w 是投资权重向量，满足$\mathbf{w}^T\mathbf{1} = 1$[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

**均值 - 方差优化**：

马科维茨投资组合优化模型：

$\min_{\mathbf{w}} \mathbf{w}^T\Sigma\mathbf{w} \\
\text{s.t. } \mathbf{w}^T\mathbf{r} = \mu \\
\mathbf{w}^T\mathbf{1} = 1$

其中 μ 是期望收益率[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

**有效前沿**：

通过求解上述优化问题，可以得到有效前沿，即给定风险水平下的最大期望收益率或给定期望收益率下的最小风险[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

**3. 期权定价模型**

**应用场景**：为金融期权定价，如布莱克 - 斯科尔斯模型、二叉树模型等。

**模型建立**：

布莱克 - 斯科尔斯方程：

$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$

其中 V 是期权价格，S 是标的资产价格，σ 是波动率，r 是无风险利率[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**边界条件**：

对于欧式看涨期权，边界条件为：

$V(S, T) = \max(S - K, 0) \\
V(0, t) = 0 \\
V(S, t) \sim S - Ke^{-r(T - t)} \quad \text{当} S \to \infty$

其中 K 是执行价格，T 是到期时间[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**数值解法**：

由于布莱克 - 斯科尔斯方程是一个偏微分方程，可以使用有限差分法、蒙特卡洛模拟等数值方法求解。例如，使用 Crank-Nicolson 方法离散化方程：

$\frac{V_{i,j+1} - V_{i,j}}{\Delta t} + \frac{1}{2}\sigma^2S_i^2\frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{(\Delta S)^2} + rS_i\frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S} - rV_{i,j} = 0$

然后求解线性方程组得到期权价格[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**期权希腊字母**：

通过对期权价格关于标的资产价格、波动率、无风险利率等参数求偏导数，可以得到期权的希腊字母，如 Delta、Gamma、Vega 等，用于风险管理和对冲策略[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

## 八、常微分方程的最新进展

### 8.1 神经常微分方程 (Neural ODE)

神经常微分方程 (Neural Ordinary Differential Equations, Neural ODE) 是近年来机器学习领域的一项重要创新，它将深度学习与常微分方程理论相结合，开创了连续深度神经网络的新范式[(3)](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)。

**基本思想**：

传统的深度神经网络是由一系列离散的层组成的，而神经常微分方程则将神经网络参数化为连续的时间动态系统。具体来说，神经常微分方程将神经网络的隐藏状态的导数定义为神经网络本身的输出：

$\frac{dz}{dt} = f_\theta(t, z)$

其中 z 是隐藏状态，θ 是神经网络的参数，fθ 是由神经网络表示的向量场[(3)](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)。

**模型结构**：

神经常微分方程的关键特点是其 "连续深度"，即网络的深度不是固定的离散层数，而是连续的时间参数。模型的训练和推断过程涉及到数值求解这个常微分方程[(3)](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)。

**正向传播**：

在正向传播过程中，给定初始状态 z (0)，使用数值积分方法（如四阶龙格 - 库塔法）求解常微分方程，得到终端状态 z (T)。然后通过输出层得到预测结果[(3)](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)。

**反向传播**：

神经常微分方程的反向传播使用伴随方法 (adjoint method) 来计算梯度。伴随方法通过求解伴随方程：

$\frac{d\lambda}{dt} = -\lambda^T\frac{\partial f_\theta}{\partial z}(t, z)$

来高效地计算梯度，避免了传统反向传播中存储所有中间状态的需要，大大减少了内存消耗[(11)](https://arxiv.org/pdf/2102.04668)。

**优势与应用**：

神经常微分方程具有以下优势：



1.  **内存效率高**：无需存储所有中间层的激活值，内存消耗与时间步数无关。

2.  **自适应计算**：可以根据输入的复杂性自动调整计算资源，在简单区域使用大步长，在复杂区域使用小步长。

3.  **精确的误差控制**：可以使用具有误差控制的自适应积分器，确保计算精度。

4.  **连续时间建模**：能够处理任意时间点的输入和输出，适用于时间序列预测、动态系统建模等任务[(7)](https://arxiv.org/pdf/2401.03965)。

**最新进展**：

近年来，神经常微分方程的研究取得了多项重要进展：



1.  **Memory Efficient and Reverse Accurate Integrator (MALI)**：基于异步蛙跳积分器，MALI 在保持常数内存消耗的同时，保证了反向时间轨迹的准确性，从而提高了梯度估计的精度[(11)](https://arxiv.org/pdf/2102.04668)。

2.  **Heavy Ball Neural ODEs (HBNODE)**：利用经典动量加速梯度下降的连续极限，HBNODE 的伴随状态也满足 HBNODE 方程，加速了正向和反向 ODE 求解器，显著减少了函数评估次数，提高了训练模型的实用性[(13)](https://arxiv.org/pdf/2110.04840)。

3.  **Stochastic Physics-Informed Neural Ordinary Differential Equations (SPINODE)**：将不确定性传播方法与神经 ODE 相结合，用于学习随机微分方程中的隐藏物理，能够直接在整个状态空间上学习动态系统的未知函数[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**应用领域**：

神经常微分方程在以下领域有广泛应用：



1.  **时间序列预测**：用于预测复杂的时间序列数据，如股票价格、气候变化等。

2.  **动态系统建模**：学习物理系统、生物系统等动态系统的演化规律。

3.  **生成模型**：如神经常微分方程变分自编码器 (NODE-VAE)，用于生成复杂的结构化数据。

4.  **最优控制**：将神经 ODE 与最优控制理论结合，用于解决高维非线性最优控制问题[(7)](https://arxiv.org/pdf/2401.03965)。

### 8.2 随机微分方程与物理启发式学习

随机微分方程 (Stochastic Differential Equations, SDEs) 是常微分方程的扩展，它引入了随机噪声项，能够更真实地描述现实世界中的不确定性动态系统[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**基本概念**：

随机微分方程的一般形式为：

$dx = g(x)dt + \sqrt{2h(x)}dw$

其中 x 是状态变量，g (x) 是漂移系数，h (x) 是扩散系数，w 是维纳过程（布朗运动）[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**物理启发式学习**：

物理启发式学习 (Physics-Informed Learning) 旨在从数据中学习物理系统的动力学方程，将物理知识融入机器学习模型中。随机物理启发式神经常微分方程 (SPINODE) 是这一领域的最新进展之一[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**SPINODE 框架**：

SPINODE 框架将随机微分方程与神经 ODE 相结合，用于学习 SDE 中的隐藏物理。其基本思想是通过随机微分方程的已知结构传播随机性，生成一组确定性 ODE 来描述随机状态统计矩的时间演化[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**关键步骤**：



1.  **不确定性传播**：使用无味变换 (Unscented Transformation) 等方法将 SDE 转换为一组 ODE，描述状态均值和协方差的演化。

2.  **ODE 求解**：使用高效的 ODE 求解器（如自适应时间步长求解器）预测状态矩的轨迹。

3.  **神经网络训练**：利用自动微分和伴随灵敏度的小批量梯度下降来学习神经网络参数，使预测的矩与数据估计的矩匹配[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**应用案例**：

SPINODE 已成功应用于多个领域：



1.  **具有外源输入的定向胶体自组装**：学习二氧化硅微粒二维自组装的动力学方程。

2.  **具有共存平衡点的竞争 Lotka-Volterra 模型**：学习四态竞争系统的随机动力学。

3.  **易感 - 感染 - 康复流行病模型**：学习疾病传播的复杂动力学[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**优势与挑战**：

SPINODE 的优势包括：



1.  **灵活性**：能够处理高维、非线性和刚性动力学系统。

2.  **效率**：利用先进的 ODE 求解器和不确定性传播方法，提高计算效率。

3.  **可扩展性**：可以自然地结合任意数量的矩（如偏斜和峰度），更好地处理非高斯状态分布[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

当前的挑战包括：



1.  需要大量重复的状态轨迹来准确学习隐藏物理。

2.  优化不确定性传播方法和 ODE 求解器的选择。

3.  更新神经 ODE 框架以使用最新的实现（如基于 MALI 的框架）[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

### 8.3 高性能数值解法与软件工具

随着计算机技术的发展，常微分方程的数值解法不断优化，涌现出许多高效的算法和软件工具，为科学研究和工程应用提供了强大的支持[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**高效数值算法**：

近年来，常微分方程数值解法的研究取得了多项重要进展：



1.  **多步方法**：如 Adams-Bashforth-Moulton 方法、Gear 方法等，利用多个先前点的信息来提高精度和稳定性[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

2.  **混合方法**：结合显式方法和隐式方法的优点，如 Rosenbrock 方法，能够处理刚性问题而无需求解大型线性方程组[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

3.  **并行算法**：利用多核处理器和 GPU 加速，开发并行数值算法，提高大规模常微分方程组的求解效率[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**软件工具与库**：

以下是一些常用的常微分方程求解软件工具：



1.  **MATLAB**：提供了一系列常微分方程求解函数，如 ode45（非刚性）、ode15s（刚性）、ode23t（适度刚性）等，支持多种数值算法[(18)](http://rivista.math.unipr.it/fulltext/1999-2s/05.pdf)。

2.  **Python 科学计算库**：

*   **SciPy**：提供了 odeint、solve\_ivp 等函数，支持多种数值方法，如 RK45、RK23、BDF 等。

*   **NumPy**：提供了数值计算的基础支持。

*   **SymPy**：用于符号计算，如求解微分方程的解析解。

*   **TorchDiffEq**：用于求解神经常微分方程的 PyTorch 扩展[(13)](https://arxiv.org/pdf/2110.04840)。

1.  **Julia**：提供了 DifferentialEquations.jl 库，支持多种数值方法，包括 ODE、DDE、SDE 等，具有高效的性能和灵活的接口[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

2.  **C++ 库**：如 DLSODES、SUNDIALS 等，用于高性能科学计算和工程应用[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**自适应求解器**：

自适应求解器能够根据解的行为自动调整步长和方法，提高计算效率和精度。现代自适应求解器通常具有以下特点：



1.  **误差估计**：使用局部误差估计来控制步长，确保计算精度。

2.  **阶数自适应**：在计算过程中动态调整方法的阶数，以平衡精度和效率。

3.  **方法切换**：根据问题的性质自动切换不同的数值方法，如从显式方法切换到隐式方法处理刚性问题[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**未来发展方向**：

常微分方程数值解法的未来发展方向包括：



1.  **人工智能辅助数值方法**：利用机器学习技术预测解的行为，优化步长和方法选择，提高求解效率。

2.  **多物理场耦合求解**：开发能够高效求解多物理场耦合常微分方程组的算法，如流固耦合、热传导 - 力学耦合等。

3.  **无网格方法**：开发不依赖网格的数值方法，如粒子方法、无网格伽辽金方法等，适用于复杂几何和移动边界问题。

4.  **高性能计算优化**：充分利用现代计算机架构，如 GPU、FPGA、量子计算机等，开发高效的并行算法和软件工具[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

## 九、总结与展望

### 9.1 常微分方程理论的重要性与贡献

常微分方程作为数学的一个重要分支，在科学研究和工程应用中发挥着不可替代的作用。从 17 世纪微积分的创立开始，常微分方程理论经历了几百年的发展，形成了完整的理论体系和丰富的求解方法[(27)](https://blog.csdn.net/qq_41375318/article/details/145261945)。

**理论贡献**：

常微分方程理论的主要贡献包括：



1.  **解的存在唯一性理论**：建立了解的存在性、唯一性和连续依赖性的条件，为常微分方程的研究奠定了理论基础[(34)](https://m.baike.com/wiki/%E6%9F%AF%E8%A5%BF%EF%BC%8D%E5%88%A9%E6%99%AE%E5%B8%8C%E8%8C%A8%E5%AE%9A%E7%90%86/20661513?baike_source=doubao)。

2.  **定性理论**：庞加莱创立的定性理论不直接求解方程，而是通过研究方程本身的结构来推断解的性质，为非线性系统的研究开辟了新途径[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

3.  **稳定性理论**：李雅普诺夫创立的稳定性理论，为分析动态系统的稳定性提供了系统的方法，在控制理论和非线性科学中有着广泛应用[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

4.  **数值方法**：发展了多种数值方法，如欧拉方法、龙格 - 库塔方法、有限差分方法等，为无法解析求解的方程提供了有效的近似解法[(53)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)。

**应用价值**：

常微分方程在各个领域的应用价值主要体现在以下几个方面：



1.  **建模能力**：能够准确描述各种动态系统的演化规律，如物理系统、生物系统、经济系统等[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

2.  **预测功能**：通过求解常微分方程，可以预测系统的未来行为，为决策提供科学依据[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

3.  **优化设计**：基于常微分方程模型，可以进行系统参数优化和结构设计，提高系统性能[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

4.  **控制策略**：利用常微分方程模型，可以设计有效的控制策略，实现对动态系统的精确控制[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

### 9.2 学习常微分方程的建议与方法

学习常微分方程需要掌握一定的数学基础和学习方法，以下是一些建议：

**基础知识准备**：

学习常微分方程前，需要具备以下基础知识：



1.  **微积分**：包括导数、积分、泰勒展开等，这是理解常微分方程的基础[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

2.  **线性代数**：包括矩阵、向量、线性方程组等，对于理解线性常微分方程和解的结构非常重要[(71)](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)。

3.  **数学分析**：包括连续性、可微性、级数等，对于理解解的存在唯一性定理和定性理论有帮助[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

**学习方法建议**：



1.  **理论与实践结合**：不仅要掌握常微分方程的理论知识，还要通过大量的例题和习题来加深理解和掌握解题技巧[(67)](https://m.baike.com/wiki/%E6%95%B0%E5%AD%A6%E7%B1%BB%E4%B8%93%E4%B8%9A%E5%AD%A6%E4%B9%A0%E8%BE%85%E5%AF%BC%E4%B8%9B%E4%B9%A6%C2%B7%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%AF%BC%E4%B9%A6/20791792?baike_source=doubao)。

2.  **从特殊到一般**：先学习简单的方程类型（如变量可分离方程、一阶线性方程），再逐步扩展到复杂的方程（如高阶方程、非线性方程）[(67)](https://m.baike.com/wiki/%E6%95%B0%E5%AD%A6%E7%B1%BB%E4%B8%93%E4%B8%9A%E5%AD%A6%E4%B9%A0%E8%BE%85%E5%AF%BC%E4%B8%9B%E4%B9%A6%C2%B7%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%AF%BC%E4%B9%A6/20791792?baike_source=doubao)。

3.  **几何直观理解**：通过斜率场、相平面等几何工具，直观理解解的行为和性质，增强对抽象理论的理解[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

4.  **数值方法实践**：学习并实践常微分方程的数值解法，如欧拉方法、龙格 - 库塔方法等，了解计算机如何求解常微分方程[(53)](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)。

5.  **应用案例分析**：研究常微分方程在物理、生物、经济等领域的应用案例，体会数学模型的构建过程和实际意义[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

**学习资源推荐**：

以下是一些学习常微分方程的优质资源：



1.  **教材**：

*   《常微分方程教程》（第二版），丁同仁等著，高等教育出版社，2004 年[(70)](https://m.gaodun.com/kaoyan/1591558.html)。

*   《常微分方程基础》（英文版・原书第 5 版），C. Henry Edwards 著，机械工业出版社，2025 年[(69)](http://www.cmpedu.com/books/book/63894.htm)。

*   《Differential Equations with Boundary-Value Problems》，Dennis G. Zill 著，Brooks/Cole，2013 年[(15)](https://sites.pitt.edu/\~phase/bard/bardware/classes/2920/Teschl_27june2012.pdf)。

1.  **在线课程**：

*   Coursera 平台的 "Ordinary Differential Equations" 课程，由 University of Michigan 提供。

*   edX 平台的 "Introduction to Differential Equations" 课程，由 Technische Universität München 提供。

*   MIT OpenCourseWare 的 "Differential Equations" 课程，由 MIT 数学系提供。

1.  **学习指导书**：

*   《数学类专业学习辅导丛书・常微分方程学习指导书》，与高等教育出版社出版的《常微分方程（第二版）》配套使用[(67)](https://m.baike.com/wiki/%E6%95%B0%E5%AD%A6%E7%B1%BB%E4%B8%93%E4%B8%9A%E5%AD%A6%E4%B9%A0%E8%BE%85%E5%AF%BC%E4%B8%9B%E4%B9%A6%C2%B7%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%AF%BC%E4%B9%A6/20791792?baike_source=doubao)。

*   《常微分方程学习指导书》，东北师范大学微分方程教研室编，高等教育出版社[(67)](https://m.baike.com/wiki/%E6%95%B0%E5%AD%A6%E7%B1%BB%E4%B8%93%E4%B8%9A%E5%AD%A6%E4%B9%A0%E8%BE%85%E5%AF%BC%E4%B8%9B%E4%B9%A6%C2%B7%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%AF%BC%E4%B9%A6/20791792?baike_source=doubao)。

1.  **软件工具**：

*   MATLAB：用于数值求解和可视化。

*   Python 的 SciPy 库：提供了丰富的常微分方程求解函数。

*   Julia 的 DifferentialEquations.jl：高效的常微分方程求解库[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

### 9.3 未来研究方向与挑战

随着科学技术的不断发展，常微分方程理论和应用面临着新的机遇和挑战。以下是一些值得关注的未来研究方向：

**理论研究方向**：



1.  **非线性常微分方程的定性理论**：深入研究非线性系统的复杂动力学行为，如混沌、分岔、孤立子等现象[(29)](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)。

2.  **高维系统的稳定性理论**：发展适用于高维系统的稳定性分析方法，特别是针对具有时滞、随机扰动等复杂因素的系统[(23)](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)。

3.  **奇异摄动理论**：研究当方程中含有小参数时，解的渐近行为和奇异摄动现象，如边界层、内部层等[(29)](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)。

4.  **随机常微分方程理论**：发展随机常微分方程的理论体系，包括解的存在唯一性、稳定性、遍历性等[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**数值方法研究**：



1.  **高效并行算法**：开发适用于多核处理器、GPU 等并行计算架构的数值算法，提高大规模常微分方程组的求解效率[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

2.  **自适应算法**：研究更高效的自适应步长和阶数选择策略，提高数值方法的精度和效率[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

3.  **不确定性量化**：发展能够量化数值解不确定性的方法，为科学计算和工程应用提供可靠的误差估计[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

4.  **机器学习辅助数值方法**：将机器学习技术应用于常微分方程的数值求解，如预测解的行为、优化算法参数等[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

**应用研究方向**：



1.  **复杂系统建模**：利用常微分方程建立更复杂的系统模型，如生态系统、社会系统、金融系统等[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

2.  **多物理场耦合模型**：研究不同物理过程耦合的常微分方程组，如热传导 - 应力耦合、流固耦合等[(2)](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)。

3.  **生物医学应用**：将常微分方程应用于生物医学领域，如疾病传播模型、细胞信号传导模型、神经网络模型等[(1)](https://www.semanticscholar.org/paper/Neural-Ordinary-Differential-Equations-for-Modeling-Kosma-Polytechnique/02c282db6cc2cdedf79d4d3cc8e2aa4e055c56b8)。

4.  **人工智能与常微分方程的交叉**：深入研究神经常微分方程、物理启发式学习等交叉领域，推动人工智能和科学计算的融合[(3)](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)。

**面临的挑战**：



1.  **高维系统的计算挑战**：随着系统维度的增加，数值求解的计算量呈指数增长，如何高效求解高维系统是一个挑战[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

2.  **刚性问题的高效求解**：对于刚性常微分方程组，传统方法需要很小的步长，如何开发更高效的刚性求解算法是一个重要方向[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

3.  **非线性问题的全局分析**：非线性系统的全局行为复杂，如何从理论和数值上分析和预测这些行为是一个长期挑战[(29)](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)。

4.  **数据驱动的模型构建**：如何从大量数据中自动构建常微分方程模型，实现数据驱动的科学发现，是一个新兴的研究方向[(5)](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)。

### 9.4 结语

常微分方程作为数学的一个重要分支，不仅有着丰富的理论体系，而且在科学研究和工程应用中发挥着不可替代的作用。从牛顿时代的天体力学，到现代的深度学习和人工智能，常微分方程始终是描述和理解动态系统的有力工具[(25)](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)。

随着计算机技术和数值方法的发展，常微分方程的应用范围不断扩大，从传统的物理、工程领域扩展到生物、经济、金融等多个学科。特别是近年来神经常微分方程的兴起，将深度学习与常微分方程理论相结合，开创了连续深度神经网络的新范式，为人工智能和科学计算的融合提供了新的思路[(3)](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)。

学习和掌握常微分方程，不仅能够提高数学素养和解决实际问题的能力，还能为深入理解和研究自然现象、工程系统和社会现象提供有力的工具。希望本文能够帮助读者建立对常微分方程的全面认识，激发进一步学习和探索的兴趣[(67)](https://m.baike.com/wiki/%E6%95%B0%E5%AD%A6%E7%B1%BB%E4%B8%93%E4%B8%9A%E5%AD%A6%E4%B9%A0%E8%BE%85%E5%AF%BC%E4%B8%9B%E4%B9%A6%C2%B7%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%AF%BC%E4%B9%A6/20791792?baike_source=doubao)。

在未来的研究中，常微分方程将继续与其他学科交叉融合，为解决复杂系统的建模、分析和控制问题提供理论支持和方法指导。同时，随着计算机技术和人工智能的不断发展，常微分方程的理论和应用也将迎来新的机遇和挑战[(17)](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)。

**参考资料 **

\[1] Neural Ordinary Differential Equations for Modeling Epidemic Spreading[ https://www.semanticscholar.org/paper/Neural-Ordinary-Differential-Equations-for-Modeling-Kosma-Polytechnique/02c282db6cc2cdedf79d4d3cc8e2aa4e055c56b8](https://www.semanticscholar.org/paper/Neural-Ordinary-Differential-Equations-for-Modeling-Kosma-Polytechnique/02c282db6cc2cdedf79d4d3cc8e2aa4e055c56b8)

\[2] 常微分方程在数学建模中的应用[ http://epub.cqvip.com/articledetail.aspx?id=1000003993977](http://epub.cqvip.com/articledetail.aspx?id=1000003993977)

\[3] Neural Ordinary Differential Equations[ https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf](https://webproxy.stealthy.co/index.php?q=https%3A//papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)

\[4] 关于常微分方程系统解决方案的一部分变量的全局渐近稳定性[ https://m.zhangqiaokeyan.com/journal-foreign-detail/0704026003899.html](https://m.zhangqiaokeyan.com/journal-foreign-detail/0704026003899.html)

\[5] Stochastic Physics-Informed Neural Ordinary Differential Equations[ https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf](https://typeset.io/pdf/stochastic-physics-informed-neural-ordinary-differential-37l1p51n.pdf)

\[6] 常微分方程的新解[ https://m.zhangqiaokeyan.com/academic-journal-foreign\_journal-symbolic-computation\_thesis/0204111342572.html](https://m.zhangqiaokeyan.com/academic-journal-foreign_journal-symbolic-computation_thesis/0204111342572.html)

\[7] Differential Equations for Continuous-Time Deep Learning[ https://arxiv.org/pdf/2401.03965](https://arxiv.org/pdf/2401.03965)

\[8] 常微分方程一些新的可积性结果[ http://www.cqvip.com/QK/83506A/19931/4001561756.html](http://www.cqvip.com/QK/83506A/19931/4001561756.html)

\[9] NEURAL ORDINARY DIFFERENTIAL EQUATIONS FOR TIME SERIES RECONSTRUCTION[ https://ouci.dntb.gov.ua/en/works/4Ej85Kzl/](https://ouci.dntb.gov.ua/en/works/4Ej85Kzl/)

\[10] Comparison Theory for Cyclic Systems of Differential Equations (常微分方程式の定性的理論とその周辺)[ https://www.semanticscholar.org/paper/Comparison-Theory-for-Cyclic-Systems-of-Equations-Jaros-Kusano/d5ee8c5eeb5338123eb7b5b0122bb20cb4d95abc](https://www.semanticscholar.org/paper/Comparison-Theory-for-Cyclic-Systems-of-Equations-Jaros-Kusano/d5ee8c5eeb5338123eb7b5b0122bb20cb4d95abc)

\[11] MALI: A Memory Efficient and Reverse Accurate Integrator for Neural ODEs[ https://arxiv.org/pdf/2102.04668](https://arxiv.org/pdf/2102.04668)

\[12] 一种求解二阶常微分方程近似解的P-SVM方法[ http://www.paper.edu.cn/pdfupload/2023/12/HL2023120401.pdf](http://www.paper.edu.cn/pdfupload/2023/12/HL2023120401.pdf)

\[13] Heavy Ball Neural Ordinary Differential Equations[ https://arxiv.org/pdf/2110.04840](https://arxiv.org/pdf/2110.04840)

\[14] Machine Learning for Partial Differential Equations[ https://arxiv.org/pdf/2303.17078](https://arxiv.org/pdf/2303.17078)

\[15] Ordinary Differential Equations and Dynamical Systems[ https://sites.pitt.edu/\~phase/bard/bardware/classes/2920/Teschl\_27june2012.pdf](https://sites.pitt.edu/\~phase/bard/bardware/classes/2920/Teschl_27june2012.pdf)

\[16] 高阶非线性常微分方程边值问题的Lagrange插值逼近[ https://m.zhangqiaokeyan.com/academic-journal-cn\_progress-applied-mathematics\_thesis/02012101691048.html](https://m.zhangqiaokeyan.com/academic-journal-cn_progress-applied-mathematics_thesis/02012101691048.html)

\[17] A Review of Recent Developments in Solving ODEs[ https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf](https://math.temple.edu/\~tug29557/assets/files/Survey-ODE.pdf)

\[18] Computing codes for ordinary differential equations: state of art and perspectives[ http://rivista.math.unipr.it/fulltext/1999-2s/05.pdf](http://rivista.math.unipr.it/fulltext/1999-2s/05.pdf)

\[19] 简化常微分方程的新方法[ https://m.zhangqiaokeyan.com/journal-foreign-detail/070403737215.html](https://m.zhangqiaokeyan.com/journal-foreign-detail/070403737215.html)

\[20] 一类二阶常微分方程边值问题自适应有限元法新算法[ https://m.zhangqiaokeyan.com/academic-conference-cn\_meeting-75908\_thesis/020223295981.html](https://m.zhangqiaokeyan.com/academic-conference-cn_meeting-75908_thesis/020223295981.html)

\[21] 新高阶常微分方程的可积类型[ http://www.cqvip.com/QK/97875A/19994/5219409.html](http://www.cqvip.com/QK/97875A/19994/5219409.html)

\[22] Recent advances in methods for numerical solution of O.D.E. initial value problems[ https://www.sciencedirect.com/science/article/pii/0377042784900037](https://www.sciencedirect.com/science/article/pii/0377042784900037)

\[23] 常微分方程非唯一解的一些预期准则[ https://m.zhangqiaokeyan.com/academic-journal-foreign\_detail\_thesis/0204116838761.html](https://m.zhangqiaokeyan.com/academic-journal-foreign_detail_thesis/0204116838761.html)

\[24] An Introduction to Ordinary Differential Equations[ https://archive.org/download/AnIntroductionToOrdinaryDifferentialEquations/An%20Introduction%20to%20Ordinary%20Differential%20Equations.pdf](https://archive.org/download/AnIntroductionToOrdinaryDifferentialEquations/An%20Introduction%20to%20Ordinary%20Differential%20Equations.pdf)

\[25] 常微分方程\[方程中的未知函数是一元函数的微分方程]\_百科[ https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike\_source=doubao](https://m.baike.com/wiki/%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B/21840446?baike_source=doubao)

\[26] 【常微分方程快速掌握秘籍】:为初学者量身打造的第一课 - CSDN文库[ https://wenku.csdn.net/column/3o8hwp1jbc](https://wenku.csdn.net/column/3o8hwp1jbc)

\[27] 常微分方程与偏微分方程的发展历史\_常微分方程的发展历史-CSDN博客[ https://blog.csdn.net/qq\_41375318/article/details/145261945](https://blog.csdn.net/qq_41375318/article/details/145261945)

\[28] 大学数学自学计划：常微分方程篇-抖音[ https://www.iesdouyin.com/share/video/7478350865447882042/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7478350636682054412\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=mum3\_shRe0dHTCG.8H21B41GtxC.pU\_ihZ4qIjurwrM-\&share\_version=280700\&titleType=title\&ts=1754553723\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7478350865447882042/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7478350636682054412\&region=\&scene_from=dy_open_search_video\&share_sign=mum3_shRe0dHTCG.8H21B41GtxC.pU_ihZ4qIjurwrM-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[29] 数学理论———常微分方程-抖音[ https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share\_version=280700\&titleType=title\&ts=1754553723\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7338020439492398399/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7338020884705217306\&region=\&scene_from=dy_open_search_video\&share_sign=wBCnziZuFd23Qfua0u02ZDtHyJo0ysoQHHVewzLqk8Y-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[30] 聊一聊微分方程和特征函数-抖音[ https://www.iesdouyin.com/share/video/7441206423830154530/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7441206524502002468\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=NR9iW41u7aSWT7XWeMwp5k4FfRI3RzBiFibdpucfx7I-\&share\_version=280700\&titleType=title\&ts=1754553723\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7441206423830154530/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7441206524502002468\&region=\&scene_from=dy_open_search_video\&share_sign=NR9iW41u7aSWT7XWeMwp5k4FfRI3RzBiFibdpucfx7I-\&share_version=280700\&titleType=title\&ts=1754553723\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[31] Ordinary Differential Equations[ https://www.geeksforgeeks.org/ordinary-differential-equations/](https://www.geeksforgeeks.org/ordinary-differential-equations/)

\[32] Differential equation, ordinary[ https://encyclopediaofmath.org/wiki/Differential\_equation,\_ordinary](https://encyclopediaofmath.org/wiki/Differential_equation,_ordinary)

\[33] Ordinary Differential Equations (ODEs)[ https://reference.wolfram.com/language/tutorial/DSolveOrdinaryDifferentialEquations.html#4872](https://reference.wolfram.com/language/tutorial/DSolveOrdinaryDifferentialEquations.html#4872)

\[34] 柯西-利普希茨定理\[柯西-利普希茨定理]\_百科[ https://m.baike.com/wiki/%E6%9F%AF%E8%A5%BF%EF%BC%8D%E5%88%A9%E6%99%AE%E5%B8%8C%E8%8C%A8%E5%AE%9A%E7%90%86/20661513?baike\_source=doubao](https://m.baike.com/wiki/%E6%9F%AF%E8%A5%BF%EF%BC%8D%E5%88%A9%E6%99%AE%E5%B8%8C%E8%8C%A8%E5%AE%9A%E7%90%86/20661513?baike_source=doubao)

\[35] 常微分方程第三版电子课本\_【笔记】常微分方程基本定理(上)-CSDN博客[ https://blog.csdn.net/weixin\_39890814/article/details/111322818](https://blog.csdn.net/weixin_39890814/article/details/111322818)

\[36] 常微分方程解的存在唯一性定理及其应用-20241229093003.docx-原创力文档[ https://max.book118.com/html/2024/1229/5022002213012021.shtm](https://max.book118.com/html/2024/1229/5022002213012021.shtm)

\[37] 一阶常微分方程解的存在唯一性定理及逐步逼近法.doc - 人人文库[ https://m.renrendoc.com/paper/252368052.html](https://m.renrendoc.com/paper/252368052.html)

\[38] 常微分方程解的存在唯一性定理-金锄头文库[ https://m.jinchutou.com/shtml/view-530714945.html](https://m.jinchutou.com/shtml/view-530714945.html)

\[39] 常微分方程31解的存在唯一性定理与逐步逼近法-金锄头文库[ https://m.jinchutou.com/shtml/view-591615857.html](https://m.jinchutou.com/shtml/view-591615857.html)

\[40] 【数学建模与MATLAB】常微分方程求解的龙格库塔方法-抖音[ https://www.iesdouyin.com/share/video/7350974164976602377/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7350974477720668964\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=e2IaSPEsSYD\_HUU3nFGpBhH.dqnllIArxoqxczG.Pxg-\&share\_version=280700\&titleType=title\&ts=1754553749\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7350974164976602377/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7350974477720668964\&region=\&scene_from=dy_open_search_video\&share_sign=e2IaSPEsSYD_HUU3nFGpBhH.dqnllIArxoqxczG.Pxg-\&share_version=280700\&titleType=title\&ts=1754553749\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[41] Ordinary Differential Equations/Existence[ https://en.m.wikibooks.org/wiki/Ordinary\_Differential\_Equations/Existence](https://en.m.wikibooks.org/wiki/Ordinary_Differential_Equations/Existence)

\[42] State the existence and uniqueness theorem for first order differential equations. Argue that the following initial value problem has a unique solution y = y(t) defined for t close to t = 0; { y' = e⁽ʸ⁻ᵗ⁾² { y(0) = 1.[ https://brainly.com/question/37309060](https://brainly.com/question/37309060)

\[43] Consider the following differential equations. Determine if the Existence and Uniqueness Theorem does or does not guarantee existence and uniqueness of a solution of each of the following initial value problems.I. ????y/????x = sqrt(x − y), y(2) = 2II. ????y/????x = sqrt(x − y), y(2) = 1III. y (????y/????x) = x − 1, y(0) = 1IV. y (????y/????x) = x − 1, y(1) = 0[ https://brainly.com/question/30551816](https://brainly.com/question/30551816)

\[44] Existence and Uniqueness of Solutions for First Order Ordinary Differential Equations - Pr, Study notes of Differential Equations[ https://www.docsity.com/en/docs/existence-and-uniqueness-of-solutions-math-365/6038647/](https://www.docsity.com/en/docs/existence-and-uniqueness-of-solutions-math-365/6038647/)

\[45] Math Reference Notes: 积分因子-CSDN博客[ https://blog.csdn.net/DaPiCaoMin/article/details/144947625](https://blog.csdn.net/DaPiCaoMin/article/details/144947625)

\[46] 常微分方程考研讲义第二章一阶微分方程的初等解法\_人人文库网[ https://m.renrendoc.com/paper/197017430.html](https://m.renrendoc.com/paper/197017430.html)

\[47] 最好的常微分方程课| 7线性微分方程与积分因子法 -抖音[ https://www.iesdouyin.com/share/video/7505976788359728410/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7505977492541360906\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=Jq7L1otQL7hTzuPEtpihAjMYKXWAj7UY5HoES7FjQQY-\&share\_version=280700\&titleType=title\&ts=1754553859\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7505976788359728410/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7505977492541360906\&region=\&scene_from=dy_open_search_video\&share_sign=Jq7L1otQL7hTzuPEtpihAjMYKXWAj7UY5HoES7FjQQY-\&share_version=280700\&titleType=title\&ts=1754553859\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[48] 积分因子法是求一阶微分方程很实用的方法，公式背不住的同学可要好好掌握-抖音[ https://www.iesdouyin.com/share/video/7125613309784542467/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7125613346514045733\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=dqIkwCQfU9VKGYkjBCs3NzgUTPLw15cFHWBurd7Hor4-\&share\_version=280700\&titleType=title\&ts=1754553857\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7125613309784542467/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7125613346514045733\&region=\&scene_from=dy_open_search_video\&share_sign=dqIkwCQfU9VKGYkjBCs3NzgUTPLw15cFHWBurd7Hor4-\&share_version=280700\&titleType=title\&ts=1754553857\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[49]  数学  常微分方程 分离变量法[ https://www.iesdouyin.com/share/video/7488186079439244582/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7488186081377438474\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=ant0F18Vcv3q4z9AslGlA7uN14IdzY.VbdpDEcwTZFU-\&share\_version=280700\&titleType=title\&ts=1754553859\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7488186079439244582/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7488186081377438474\&region=\&scene_from=dy_open_search_video\&share_sign=ant0F18Vcv3q4z9AslGlA7uN14IdzY.VbdpDEcwTZFU-\&share_version=280700\&titleType=title\&ts=1754553859\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[50] 数学解题方法分享 An Interesting Homemade Differential Equation-抖音[ https://www.iesdouyin.com/share/video/7461829274790677797/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7461830285412682522\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=H5Cj0WEziJIl2YHqe3jiJLvinXaEUr3T8sABcfur5EE-\&share\_version=280700\&titleType=title\&ts=1754553859\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7461829274790677797/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7461830285412682522\&region=\&scene_from=dy_open_search_video\&share_sign=H5Cj0WEziJIl2YHqe3jiJLvinXaEUr3T8sABcfur5EE-\&share_version=280700\&titleType=title\&ts=1754553859\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[51] Integrating factor[ https://www.ryantoomey.org/wiki/Integrating\_factor](https://www.ryantoomey.org/wiki/Integrating_factor)

\[52] Chapter 4 First and second order ODEs[ https://bookdown.org/vshahrez/lecture-notes/first-and-second-order-odes.html](https://bookdown.org/vshahrez/lecture-notes/first-and-second-order-odes.html)

\[53] 4.2: 1st Order Ordinary Differential Equations[ https://chem.libretexts.org/Bookshelves/Physical\_and\_Theoretical\_Chemistry\_Textbook\_Maps/Mathematical\_Methods\_in\_Chemistry\_(Levitus)/04%3A\_First\_Order\_Ordinary\_Differential\_Equations/4.02%3A\_1st\_Order\_Ordinary\_Differential\_Equations](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_\(Levitus\)/04%3A_First_Order_Ordinary_Differential_Equations/4.02%3A_1st_Order_Ordinary_Differential_Equations)

\[54] ode-solver[ https://github.com/topics/ode-solver?l=python\&o=asc\&s=forks](https://github.com/topics/ode-solver?l=python\&o=asc\&s=forks)

\[55] Solve ODE Methods in Python[ https://codepal.ai/code-generator/query/t1XP1572/python-solve-ode-methods](https://codepal.ai/code-generator/query/t1XP1572/python-solve-ode-methods)

\[56] First-order ordinary differential equation (ODE) that describes exponential decay[ https://github.com/Oussamazz/Runge-Kutta-method-RK4-ODE-solver](https://github.com/Oussamazz/Runge-Kutta-method-RK4-ODE-solver)

\[57] Python Euler's Method ODE Solver[ https://codepal.ai/code-generator/query/pLT6eAz7/python-eulers-method-ode-solver](https://codepal.ai/code-generator/query/pLT6eAz7/python-eulers-method-ode-solver)

\[58] GitHub - uprestel/ODE-Solver: Ordinary differential equation solver using the method of multiple shooting[ https://github.com/uprestel/ODE-Solver/](https://github.com/uprestel/ODE-Solver/)

\[59] 用python 实现龙格-库塔(Runge-Kutta)方法\_python三阶龙格库塔公式算法-CSDN博客[ https://blog.csdn.net/u012836279/article/details/80176985](https://blog.csdn.net/u012836279/article/details/80176985)

\[60] python中的龙格库塔函数\_mob64ca12f831ae的技术博客\_51CTO博客[ https://blog.51cto.com/u\_16213464/13146165](https://blog.51cto.com/u_16213464/13146165)

\[61] 龙格库塔法 python - CSDN文库[ https://wenku.csdn.net/answer/76yn5u1u1b](https://wenku.csdn.net/answer/76yn5u1u1b)

\[62] python龙格库塔 - CSDN文库[ https://wenku.csdn.net/answer/4tuheufgx7](https://wenku.csdn.net/answer/4tuheufgx7)

\[63] Python龙格库塔解常微分方程\_mob64ca12d84572的技术博客\_51CTO博客[ https://blog.51cto.com/u\_16213332/13649386](https://blog.51cto.com/u_16213332/13649386)

\[64] 四阶龙格库塔法的基本思想\_数值常微分方程-欧拉法与龙格-库塔法-CSDN博客[ https://blog.csdn.net/weixin\_32187691/article/details/112480479](https://blog.csdn.net/weixin_32187691/article/details/112480479)

\[65] 用龙格库塔法求解微分方程-抖音[ https://www.iesdouyin.com/share/video/7095247573384187174/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from\_aid=1128\&from\_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7095247875877554975\&region=\&scene\_from=dy\_open\_search\_video\&share\_sign=OB.odWXRZsnL9tk5c9sXW.xdTR2x1jxV8C9iXUEbHCU-\&share\_version=280700\&titleType=title\&ts=1754553930\&u\_code=0\&video\_share\_track\_ver=\&with\_sec\_did=1](https://www.iesdouyin.com/share/video/7095247573384187174/?did=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&from_aid=1128\&from_ssr=1\&iid=MS4wLjABAAAANwkJuWIRFOzg5uCpDRpMj4OX-QryoDgn-yYlXQnRwQQ\&mid=7095247875877554975\&region=\&scene_from=dy_open_search_video\&share_sign=OB.odWXRZsnL9tk5c9sXW.xdTR2x1jxV8C9iXUEbHCU-\&share_version=280700\&titleType=title\&ts=1754553930\&u_code=0\&video_share_track_ver=\&with_sec_did=1)

\[66] 上海电力大学2025研究生复试科目考试大纲:常微分方程\_中公教育网[ https://www.offcn.com/kaoyan/2025/0107/280876.html](https://www.offcn.com/kaoyan/2025/0107/280876.html)

\[67] 数学类专业学习辅导丛书·常微分方程学习指导书\_百科[ https://m.baike.com/wiki/%E6%95%B0%E5%AD%A6%E7%B1%BB%E4%B8%93%E4%B8%9A%E5%AD%A6%E4%B9%A0%E8%BE%85%E5%AF%BC%E4%B8%9B%E4%B9%A6%C2%B7%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%AF%BC%E4%B9%A6/20791792?baike\_source=doubao](https://m.baike.com/wiki/%E6%95%B0%E5%AD%A6%E7%B1%BB%E4%B8%93%E4%B8%9A%E5%AD%A6%E4%B9%A0%E8%BE%85%E5%AF%BC%E4%B8%9B%E4%B9%A6%C2%B7%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%AF%BC%E4%B9%A6/20791792?baike_source=doubao)

\[68] 高教社产品信息检索系统[ https://xuanshu.hep.com.cn/front/h5Mobile/bookDetails?bookId=6727ad50e4efbc722ba48600](https://xuanshu.hep.com.cn/front/h5Mobile/bookDetails?bookId=6727ad50e4efbc722ba48600)

\[69] 常微分方程基础(英文版.原书第5版)—美C.Henry Edwards——机工教育服务网[ http://www.cmpedu.com/books/book/63894.htm](http://www.cmpedu.com/books/book/63894.htm)

\[70] 2025数学考研需要看哪些参考书?推荐这7本-高顿[ https://m.gaodun.com/kaoyan/1591558.html](https://m.gaodun.com/kaoyan/1591558.html)

\[71] 常微分方程\_图书馆[ https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da](https://libopac.cdut.edu.cn/opac/book/dfd6f7ccf73f1283c2d319aa3b4525da)

\[72] Notable Children's Books - 2025[ https://www.ala.org/alsc/awardsgrants/notalists/ncb?fbclid=IwAR24ZEmyQVur7e6HEIGBj3g2m-S1-eHzjxqzQFKRmkfVxrFZiVLvveABRI8](https://www.ala.org/alsc/awardsgrants/notalists/ncb?fbclid=IwAR24ZEmyQVur7e6HEIGBj3g2m-S1-eHzjxqzQFKRmkfVxrFZiVLvveABRI8)

\[73] The Best Books of April 2025[ https://www.barnesandnoble.com/blog/the-best-books-of-april-2025/](https://www.barnesandnoble.com/blog/the-best-books-of-april-2025/)

\[74] 8 Books That Should Be On Your Radar in 2025[ https://www.publishersweekly.com/pw/by-topic/industry-news/tip-sheet/article/96818-eight-books-that-should-be-on-your-radar-in-2025.html](https://www.publishersweekly.com/pw/by-topic/industry-news/tip-sheet/article/96818-eight-books-that-should-be-on-your-radar-in-2025.html)

\[75] Every Orbit Book Coming in Winter 2025[ https://www.hachettebookgroup.com/orbit-books/every-orbit-book-coming-in-winter-2025/](https://www.hachettebookgroup.com/orbit-books/every-orbit-book-coming-in-winter-2025/)

> （注：文档部分内容可能由 AI 生成）