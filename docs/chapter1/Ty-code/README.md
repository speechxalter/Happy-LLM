# 用 NumPy 从零实现 CBOW

## 1. CBOW 的基本思想

CBOW（Continuous Bag of Words）把中心词附近的词向量平均起来，用平均向量预测中心词。以第一句 *The quick brown fox jumps over the lazy dog.* 为例，中心词是 **brown**，输入是 **the、quick、fox、jumps**。四个词的排列顺序可以改变，平均向量及预测结果保持相同。若某个词出现多次，它的每次出现都参与平均。

| 符号 | 本实验取值 | 含义 |
| --- | ---: | --- |
| Window Size | 2 | 从中心位置向左、向右各最多取两个词 |
| Embedding Dimension $d$ | 100 | 每个词向量的坐标数 |
| Vocabulary Size $V$ | 25 | 整份语料中不同词的数量，也是输出类别数 |
| Context Length $C$ | 主要样本为 4 | 当前样本实际取到的上下文词数，包含重复出现 |

**代码** `preprocess_corpus()` 将每行作为独立句子，转为小写并提取连续的英文字母。`build_vocab()` 按词首次出现的顺序分配 ID。`generate_training_data()` 遍历各句的中心位置，裁剪窗口边界并排除中心位置。例如，第一句开头的 `the` 只有 `quick` 和 `brown` 两个上下文词。八个句子的 65 个词位置各产生一条训练样本。

## 2. 为什么使用上下文预测中心词

上下文提供了中心词可能是什么的线索。为了提高 `brown` 的预测概率，模型会调整输出权重，也会调整 `the`、`quick`、`fox`、`jumps` 的向量。一个词的向量在不同样本中反复被使用，逐步积累有利于预测中心词的信息。

CBOW 学的是 $P(w_t \mid \text{中心位置左右两侧的上下文})$。在这个例子里，`fox` 和 `jumps` 位于 `brown` 之后，仍参与预测。用于自回归语言建模的 $n$-gram 用历史词 $w_{t-n+1},\ldots,w_{t-1}$ 预测下一个词 $w_t$。例如，trigram 预测 `brown` 时只看到 `the quick`，并且保留它们的顺序。

**代码** 在 `generate_training_data()` 中查看中心位置前后的两个切片，再到 `CBOW.forward()` 看 `mean(axis=0)` 如何沿位置维度聚合。

## 3. One-Hot 与 Embedding Lookup 的关系

词 ID 为 $i$ 时，令 $e_i\in\mathbb{R}^{1\times V}$ 为只有第 $i$ 个坐标等于 1 的行向量。乘法 $e_i W_{\mathrm{in}}$ 恰好选出 $W_{\mathrm{in}}$ 的第 $i$ 行。将 $C$ 个 one-hot 向量堆成矩阵 $X$，便得到

$$
XW_{\mathrm{in}}
=
\begin{bmatrix}
W_{\mathrm{in}}[c_1,:]\\
\vdots\\
W_{\mathrm{in}}[c_C,:]
\end{bmatrix}
\in\mathbb{R}^{C\times d}.
$$

这里 $c_j$ 表示上下文第 $j$ 个位置的词 ID。示例的上下文 ID 是 `[0, 1, 3, 4]`，目标 `brown` 的 ID 是 `2`。$(4,25)$ 的 one-hot 矩阵乘以 $(25,100)$ 的输入嵌入矩阵，结果为 $(4,100)$。

**代码** `show_example()` 构造 one-hot 矩阵，验证矩阵乘法和查表所得结果逐元素相同。训练中的 `CBOW.forward()` 直接使用 `self.W_in[context_ids, :]` 取行。

## 4. 输入嵌入矩阵的含义

$W_{\mathrm{in}}\in\mathbb{R}^{V\times d}$ 的每一行对应一个词，每一列对应词向量的一个坐标。词 ID $i$ 确定行号，行内的 $d$ 个浮点数是随训练更新的参数。只要词 $i$ 在某个样本的上下文中出现，它的输入向量就会从该样本收到梯度。中心词 ID 用于指定输出层的正确类别。

**代码** `initialize_parameters()` 随机初始化 `W_in` 和 `W_out`，两者元素从均值为 0、标准差为 $1/\sqrt{d}$ 的正态分布采样。输出偏置 `b` 从零开始。

## 5. 为什么取上下文词向量的平均值

输出层接收一个 $d$ 维向量，而句首、句尾的样本通常只有两个或三个上下文词。沿上下文维度求平均，可以把不同长度的上下文都变成一个 $d$ 维向量。除以实际词数 $C$ 让各上下文词等权，避免仅因窗口里有更多词就累加更大的权重。

**代码** `CBOW.forward()` 中，`context_vectors.mean(axis=0)` 对 $(C,d)$ 的每一列求均值。

## 6. 隐藏层的数学表示

令 $v_j=W_{\mathrm{in}}[c_j,:]$ 表示上下文第 $j$ 个位置的词向量。隐藏表示为

$$
h=\frac{1}{C}\sum_{j=1}^{C} v_j
=\frac{v_{\mathrm{the}}+v_{\mathrm{quick}}+v_{\mathrm{fox}}+v_{\mathrm{jumps}}}{4}
\quad\text{（本例）}.
$$

公式里把 $h$ 写成行向量。NumPy 中 `h` 是形状为 `(100,)` 的一维数组。训练前，本例的前四个坐标约为 `[0.03686, -0.02695, 0.05196, 0.02390]`。

**代码** `CBOW.forward()` 生成 `h`，`show_example()` 打印它的形状和部分数值。

## 7. 输出层如何给词汇表中的词打分

$$
z=hW_{\mathrm{out}}+b,\qquad
z_j=\sum_{i=1}^{d} h_i(W_{\mathrm{out}})_{ij}+b_j.
$$

- $W_{\mathrm{out}}\in\mathbb{R}^{d\times V}$ 的第 $j$ 列用于给候选词 $j$ 打分。
- $b\in\mathbb{R}^{V}$ 的第 $j$ 个元素是候选词 $j$ 的可训练偏置。
- $z\in\mathbb{R}^{V}$ 是整张词汇表的原始分数（logits）。本例中，`(100,) @ (100, 25) + (25,)` 得到 `(25,)`。

**代码** `CBOW.forward()` 中的 `logits = h @ self.W_out + self.b`。随后的完整 softmax 把这 25 个分数转换成一个概率分布。

## 8. Softmax 与 Cross Entropy Loss

$$
p_j=\frac{\exp(z_j)}{\sum_{k=1}^{V}\exp(z_k)},\qquad
\sum_{j=1}^{V}p_j=1.
$$

设 $t$ 是目标词 ID，$y$ 是它的 one-hot 向量。单样本交叉熵为

$$
L=-\sum_{j=1}^{V}y_j\log p_j=-\log p_t.
$$

所有 logits 同减 $m=\max_j z_j$，分子与分母同乘 $e^{-m}$，概率保持相同。这样指数的输入最大为 0。代码先计算 $\log p_j=(z_j-m)-\log\sum_k e^{z_k-m}$，再取指数得到 $p_j$。直接使用 $\log p_t$ 计算损失，可以避免极小概率在浮点数中变成 0 后再取对数。

本例初始的 $z_{\mathrm{brown}}\approx -0.024883$，$p_{\mathrm{brown}}\approx 0.038962$，所以 $L\approx 3.245167$。这是单条样本的损失。

**代码** `CBOW.forward()` 中的 `shifted`、`log_probs` 和 `probs` 对应上述计算。`compute_loss()` 通过 `-log_probs[target_id]` 取出交叉熵。

## 9. 反向传播的完整推导

以下梯度都从同一次前向传播得到。在 NumPy 中，`dz`、`db`、`dh` 都以一维数组表示。

### 9.1 从损失到 logits

将 softmax 代入交叉熵，再对第 $j$ 个分数求导。

$$
L=-z_t+\log\sum_{k=1}^{V}e^{z_k},\qquad
\frac{\partial L}{\partial z_j}
=-\mathbf{1}[j=t]+\frac{e^{z_j}}{\sum_k e^{z_k}}
=p_j-y_j.
$$

$g_z=\nabla_z L=p-y$，形状为 `(V,)`。本例中 $g_z[\mathrm{brown}]\approx -0.961038$，其余词的分量等于各自的预测概率。

**代码** `CBOW.backward()` 复制 `result.probs`，然后执行 `dz[target_id] -= 1.0`。

### 9.2 从 logits 到输出权重、偏置和隐藏层

由 $z_j=\sum_i h_i(W_{\mathrm{out}})_{ij}+b_j$ 得

$$
\frac{\partial L}{\partial (W_{\mathrm{out}})_{ij}}
=g_{z,j} h_i,\qquad
\frac{\partial L}{\partial b_j}=g_{z,j}.
$$

所以 $\nabla_{W_{\mathrm{out}}} L=h^\top g_z$ 是形状为 `(d, V)` 的外积，$\nabla_b L=g_z$ 的形状为 `(V,)`。每个 $h_i$ 都参与计算全部 $V$ 个输出分数，要把各条路径的贡献加起来。

$$
\frac{\partial L}{\partial h_i}
=\sum_{j=1}^{V}g_{z,j}(W_{\mathrm{out}})_{ij},\qquad
g_h=\nabla_h L=g_z W_{\mathrm{out}}^\top.
$$

**代码** `np.outer(result.h, dz)` 得到 `dW_out`，`db = dz` 得到偏置梯度，`dz @ self.W_out.T` 得到形状为 `(d,)` 的 `dh`。

### 9.3 从隐藏层到输入嵌入矩阵

隐藏层每个坐标都是对应上下文坐标的平均值。对任意上下文位置 $j$，都有

$$
\frac{\partial h_i}{\partial v_{j,i}}=\frac{1}{C},
\qquad \nabla_{v_j}L=\frac{g_h}{C}.
$$

查表时每次出现都从 $W_{\mathrm{in}}$ 取得一行，反传时就沿相同索引把梯度加回去。因此第 $r$ 行的梯度为

$$
\nabla_{W_{\mathrm{in}}[r,:]} L
=\sum_{j=1}^{C}\mathbf{1}[c_j=r]\frac{g_h}{C}.
$$

本例中四个上下文词各出现一次，对应四行各加 $g_h/4$。`brown` 的输入嵌入行在本次反传中的梯度为零。如果一个词出现两次，对应行就加两次。位置数 $C$ 随样本变化，所以句首只有两词时每次出现收到 $g_h/2$。

**代码** `CBOW.backward()` 创建零矩阵 `dW_in`，用循环执行 `dW_in[word_id] += per_context_gradient`。每次执行都保留同一行已经累计的梯度。

### 9.4 用中心差分核对手算梯度

每次只扰动一个标量参数 $\theta_i$。在两侧重新前向传播，比较中心差分与反传计算的对应项。

$$
g_i^{\mathrm{num}}
\approx \frac{L(\theta_i+\varepsilon)-L(\theta_i-\varepsilon)}{2\varepsilon},
\qquad \varepsilon=10^{-5}.
$$

**代码** `gradient_check()` 从 `W_in`、`W_out`、`b` 抽查 7 项，每项检查后恢复原值。比如 `W_in[0, 0]` 的手算和数值梯度都约为 $-0.01506209$，`W_in[2, 0]` 在本例中两者均为 0。本次最大绝对误差为 $1.40\times 10^{-11}$。

## 10. 参数更新

训练使用逐样本随机梯度下降（SGD）。每轮打乱全部 65 条样本，每条样本依次做前向传播、计算梯度、更新参数。令 $\eta=0.1$，更新式为

$$
\begin{aligned}
W_{\mathrm{in}} &\leftarrow W_{\mathrm{in}}-\eta\,\nabla_{W_{\mathrm{in}}}L,\\
W_{\mathrm{out}} &\leftarrow W_{\mathrm{out}}-\eta\,\nabla_{W_{\mathrm{out}}}L,\\
b &\leftarrow b-\eta\,\nabla_b L.
\end{aligned}
$$

**代码** `train()` 在做三项更新之前先算完梯度。初始时和每轮结束后，`mean_loss()` 都用当时固定的参数重新计算全部样本的平均交叉熵。返回的 `loss_history[0]` 是训练前的损失，`loss_history[e]` 是第 $e$ 轮结束后的损失。终端打印第 1 轮、每 25 轮和最后一轮的记录。

## 11. 提取训练后的词向量

`model.W_in[word_to_id["brown"]]` 取出训练后的 `brown` 向量，形状为 `(100,)`。若要比较它与另一个词的向量方向，可以计算

$$
\operatorname{cos}(u,v)=\frac{u\cdot v}{\lVert u\rVert_2\lVert v\rVert_2}.
$$

**代码** `cosine_similarity()` 计算两个非零向量的余弦相似度。`main()` 逐词计算得分，排除 `brown` 自身并按降序取前五。小规模人工语料上的相似度适合观察训练机制。要把相似度解释为可靠的语言学结论，还需要更有代表性的语料和实证检验。

## 运行与实测结果

`corpus.txt` 保存八句语料，`cbow_numpy.py` 包含数据处理、模型、梯度检查和运行入口。脚本按自身所在的目录读取语料。依赖为 Python 和 NumPy。

```powershell
Set-Location -LiteralPath 'E:\分支2：工作\搜广推算法\1. DataWhale Happy-LLM\Happy-LLM\docs\chapter1\Ty-code'
python -B .\cbow_numpy.py
```

实测环境为 Python 3.12.7 和 NumPy 1.26.4，随机种子为 `42`，训练 150 轮。

| 轮数 | 全语料平均损失 |
| ---: | ---: |
| 0 | 3.214426 |
| 1 | 2.954791 |
| 25 | 0.691364 |
| 50 | 0.301702 |
| 75 | 0.248848 |
| 100 | 0.234629 |
| 125 | 0.218934 |
| 150 | 0.208825 |

训练结束后，`model.predict(["the", "quick", "fox", "jumps"], top_k=5)` 给出以下训练样本内的预测概率。每个概率都基于完整词汇表。

| 排名 | 候选词 | 概率 |
| ---: | --- | ---: |
| 1 | brown | 0.969167 |
| 2 | red | 0.018097 |
| 3 | over | 0.011481 |
| 4 | behind | 0.000308 |
| 5 | slow | 0.000271 |

与 `brown` 的输入词向量最相似的五个词依次为 `red`（0.394320）、`house`（0.251189）、`field`（0.205819）、`lazy`（0.169172）、`river`（0.146971）。

## 阅读顺序

1. 用 `preprocess_corpus()`、`build_vocab()` 和 `generate_training_data()` 从第一句手写出 `brown` 样本，再找出第一个 `the` 的边界样本。
2. 跟着 `show_example()` 阅读 `CBOW.forward()` 和 `compute_loss()`，在纸上写出 `(4, 100) → (100,) → (25,) → 标量损失`。
3. 重点读 `CBOW.backward()`。沿着 `dz → dW_out / db / dh → dW_in` 对照第 9 节，再用 `gradient_check()` 跟踪 `W_in[0, 0]` 这一项。
4. 读 `train()` 中一次 SGD 更新和每轮的损失记录，最后用 `CBOW.predict()` 和 `cosine_similarity()` 看同一个样本经过训练后的输出。
