# flint

**一个在本地做决策、不写文字的模型。**

[English](README.md) · [简体中文](README.zh.md)

给它一段上下文和一小撮封闭选项,它返回证据指向的那一项,外加一个可以被追责的置信度。它从不生成文字,所以也没法编造文字;在一台笔记本上,它是几十毫秒。

它存在的理由,是应用每时每刻都在做、又送不出去的那些判断:**用户指的是这几个候选里的哪一个、这个字段该填哪个值、这个状态之后该做哪个动作。** 选择与偏好——在已经存在的东西里挑——而不是生成。本地几十毫秒出结果的,才可能在每一次按键上问它;一秒出结果的,一次都问不起。

这个仓库是配方、评测协议和服务契约。**架构不是我们的,也不在这里重新实现。**

## 为什么它是另一种形状的模型

| | 聊天模型 | Flint |
| --- | --- | --- |
| 输出 | 任意形状的文字 | 封闭集合里的一个选项 |
| 失败形态 | 一段流畅的错话 | 一个错的选项,附带概率 |
| 开销 | 按 token 计 | 一次前向,零 token |
| 跑在哪 | 别处 | 用户这台机器上 |
| 你拿它搭什么 | 一个 prompt | 一个 `if` |

要紧的是第二行。生成的答案是「对」或「错」,中间没有任何信号;而一个决策带着概率,调用方可以自己定阈值,高于它就用、低于它就弃权。这个模型之所以是现在这个形状,原因全在这里——阈值买到了什么、代价是什么,`docs/evaluation.md` 里有实测。

## 哪些是我们的,哪些不是

Flint 在别人发布的编码器上训练一个决策头,用的是别人造的引擎。

- **我们的:** 一份标注语料怎么变成决策记录(`flint/scenarios.py`);训练混合集怎么组装并审计,好让它确实是配方声称的那一份(`flint/mixture.py`);评测该量什么、为什么(`flint/metrics.py`、`docs/evaluation.md`);推理与服务链路;以及检查点。
- **不是我们的:** 架构。`flint/engine.py` 是唯一与它对话的模块,而且只是一层薄适配——引擎是一个公开的 Apache-2.0 包,`docs/method.md` 记录了它的设计出处和文献。

引擎的任何部分都没有被抄进这个仓库。想理解模型本身,去读引擎;想理解**怎么训出一个对自己的置信度诚实的模型**,读 `flint/metrics.py`。

## 目录

```
flint/metrics.py        误差预算下的覆盖率、校准、顺序敏感性
flint/scenarios.py      语料 -> 决策记录,选项顺序被打乱
flint/mixture.py        命名来源、去重、逐行溯源、审计
flint/engine.py         唯一的上游决策引擎适配层
flint/train.py          训练过程,以及操作点拟合
flint/evaluate.py       评测报告
flint/inference.py      检查点布局、加载、作答
flint/serve.py          POST /v1/systemone(只用标准库)
flint/smoke.py          全链路、离线、几秒跑完
docs/method.md          架构,以及它背后的文献
docs/training.md        配方,以及哪些语料是能用的
docs/evaluation.md      该量什么,以及它会在哪三处骗你
docs/api.md             请求与响应格式
docs/licensing.md       MIT 授予覆盖到哪、没覆盖到什么
```

## 快速开始

需要 Python 3.10+。

```bash
pip install -e .          # 会拉入引擎(Apache-2.0)、torch、transformers

# 不下载任何东西,先证明整条链路能跑:
python -m flint.smoke
```

`flint.smoke` 在本地造一个极小的编码器和分词器,合成一批决策记录,训练几步,写出检查点,再用**服务端实际使用的那个加载器**把它读回来并回答一个问题。几秒钟,不需要网络。它证明的是管道通,不是模型好——64 维随机初始化的编码器做出来的东西不可能好用。

一次真实运行:

```bash
flint-build --spec spec.json --out data/mix.jsonl
flint-train --mix data/mix.jsonl --dev data/dev.jsonl --out runs/flint-0.1.0 --device mps
flint-evaluate --checkpoint runs/flint-0.1.0 --suite data/dev.jsonl --mix data/mix.jsonl
```

服务:

```bash
flint-serve --checkpoint runs/flint-0.1.0 --port 8080 --threshold 0.5
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Order 4471 shipped on the 3rd. The customer asked to return it on the 9th.",
  "questions": {"pick": {"type": "choice",
                         "instructions": "which option does the state describe?",
                         "criteria": {"c1": "inside the return window",
                                      "c2": "outside the return window"}}}}'
```

```json
{"pick": {"choice": "c2", "probabilities": {"c1": 0.31, "c2": 0.69}, "confidence": 0.19,
          "release": false}}
```

## 现有的数字

在已发布的检查点上实测——322M 参数、fp16、Apple M4 Pro。写清机器,是因为**不讲机器的延迟只是传闻**;写清版本,是因为**不讲版本的数字下个月就是另一个模型**。

| 量 | 值 | 怎么测的 |
| --- | --- | --- |
| 一次 16 选一的决策 | **24.8 ms** p50,25.1 ms p95 | MLX,预热后,单请求,15 次采样 |
| 各套件里的一题 | p50 10.3 ms,p95 58.9 ms | 1,380 题,选项数按各套件原样 |
| 模型占用磁盘 | 643,835,524 字节 | fp16,CPU 或 GPU,不需要量化 |

准确率不是一个数,印一个数出来恰恰是这个仓库想避免的那种不诚实。**任务像它训过的东西时它有用,不像的时候它接近瞎猜:**

| 套件 | n | 准确率 | |
| --- | ---: | ---: | --- |
| 智能粘贴,域内 | 200 | **78.5%** | 它被造出来干的活 |
| 拼音候选,域外 | 96 | 67.7% | 相邻任务,没见过的说法 |
| 短续写,域外 | 48 | 56.2% | 相邻任务,没见过的说法 |
| 智能粘贴,域外 | 192 | 34.4% | 同一任务,没见过的 state |
| MMLU | 116 | 23.3% | 抽象推理,4 选项 |
| MMLU-Pro | 200 | 13.5% | 抽象推理,10 选项 |

**域外那几行才是诚实的部分。** state 变了准确率还纹丝不动的决策模型,不是这个东西——state 就是输入,没见过的 state 就是没见过的任务。

它真正的长处不是准确率,而是**弃权**——知道自己不知道。在这条轴上,产出这个检查点的那次配方修正,比准确率值钱:

| | 配对置信度落差 | 变得更不自信 | 对照集准确率 |
| --- | ---: | ---: | ---: |
| 上一版配方 | −0.043 | 34.5% | 34.5% |
| **本检查点** | **+0.041** | **62.7%** | **60.9%** |

最后一行要照实读:这是在**混合集覆盖到的那几个族**上量的,落差 +0.050;而在**不在混合集里的那两个族**上是 −0.002——**这个改进并不迁移**。`docs/evaluation.md` §6 写了这个量法,那是本仓库最该先读的一段。

我们**没有**的:任何公开榜单数字。这不是一个小号聊天模型,没有哪个榜能把它放进去——它的任务是「在这几个选项里挑」,唯一诚实的量法是拿一份你能逐条检查的套件去量。`docs/evaluation.md` 讲怎么造这样一份套件,以及怎么防止它说谎。

## 已发布的模型

**`talex-flint-1.0`** —— 322M 参数、fp16、Apache-2.0。

[**下载**](https://github.com/talex-touch/flint/releases/tag/talex-flint-1.0) —— release 正文就是模型卡,里面有压缩包的 sha256。

```bash
# MLX,上面的延迟就是用它量的
pip install laya-mlx
python -c "import laya_mlx; a = laya_mlx.load('talex-flint-1.0'); print(a.predict('Order 4471 shipped.', {'q': {'type': 'noul', 'instructions': 'was it shipped?'}}))"

# PyTorch,走本仓库
python -c "
from flint.inference import FlintCheckpoint
c = FlintCheckpoint.load('talex-flint-1.0')
print(c.answer([{'state': 'Order 4471 shipped.', 'questions': {'q': {'type': 'noul', 'instructions': 'was it shipped?'}}}])[0])"
```

两条路加载的是同一个目录,而且**互相校验过**:在 200 条留出的粘贴决策上,两者选项一致 **200/200**。

权重不提交进本仓库——643 MB 不该进 git。它也不是另一样产品:压缩包里就是这套配方写出来的检查点,而 `flint-train` 会用同样的方式、从一份你担得起责任的混合集里再写出一个。**为什么权重是 Apache-2.0 而代码是 MIT**,以及这条界线覆盖到什么,记在 `docs/licensing.md`。

## 许可

| | |
| --- | --- |
| 代码、schema、文档 | **MIT** —— [`LICENSE`](LICENSE) |
| 已发布的权重(`talex-flint-1.0`) | **Apache-2.0** |

两个许可,因为它们是两样东西。权重是两个宽松许可上游的衍生作品——`laya` 引擎及其已发布检查点(Apache-2.0),底座是 `jhu-clsp/mmBERT-base`(MIT)——而 Apache-2.0 是能把这些许可要求的声明义务**一路带下去**的那一个。本仓库自己写的代码没有这个义务,保持 MIT。

[`docs/licensing.md`](docs/licensing.md) 写的是最容易搞错的那部分:**换一个底座**会对上面这个答案做什么,以及为什么在一个你无权再分发的语料上训出来的检查点,是一个**每个下载者都会静默继承**的许可问题。`docs/training.md` §2 记录了那几个「显而易见的选择」语料实际查到的许可——它们大多不允许再分发,这也是**这批权重背后的混合集是合成出来的**原因。
