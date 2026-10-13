# ME5411 Robot Vision and AI Group Project 要求详解

> 来源：`ME5411_Project_AY2627.pdf`（AY2026/27）  
> 本文将 PDF 中明确写出的要求与执行建议分开整理。标记为“必须”的内容属于原文硬性要求；“建议”部分是为了降低漏项风险而补充的执行方法，不代表 PDF 新增要求。

## 1. 最重要信息速览

| 项目 | 明确要求 |
|---|---|
| 项目形式 | Group Project（小组项目），software-based project |
| 截止时间 | **2026 年 11 月 20 日 23:59（Singapore time）** |
| 提交位置 | CANVAS 的 **Assignments** 区域下的 **Project** |
| 提交内容 | 一份项目报告，以及对应的 MATLAB 或 Python implementation code |
| 提交压缩包 | `Group_XX.zip`，其中 `XX` 为两位小组编号，例如第 1 组写作 `Group_01.zip` |
| 编程语言 | **MATLAB 或 Python，二选一** |
| 报告格式 | Arial，11 point，single-line spacing，最多 50 页；典型报告为 30–40 页 |
| 数据划分 | `dataset_2026.zip` 中的数据按 **75% training set / 25% test set** 划分 |
| 核心比较 | 设计一个 CNN 分类方法和一个 non-CNN 分类方法，并比较两者的 effectiveness 与 efficiency |
| 小组成绩 | 项目根据报告和代码评估；同组所有成员获得相同分数 |

## 2. 项目总体目标（Objective）

本项目的目标是在给定的 microchip image（微芯片图像）上：

1. 设计（design）不同的 image-processing 与 machine-learning 方法；
2. 实现（implement）这些方法；
3. 比较（compare）不同方法；
4. 最终识别 microchip image 中的字符。

PDF 的 Figure 1 展示了目标芯片标签图像。原始图像文件名为 `charact2.bmp`，可从 CANVAS 获取。

## 3. 七项任务逐项说明（Tasks）

PDF 明确要求完成以下 **全部七项任务**。其中：

- Task 1–6 对应课程 Part 1；
- Task 7 对应课程 Part 2。

### Task 1：显示原图并进行对比度增强

必须完成：

1. 在屏幕上显示原始图像；
2. 对图像进行 contrast enhancement（对比度增强）实验；
3. 对实验结果作出评论和分析。

这里不仅需要给出处理后的图片，还需要讨论增强前后的效果，例如字符与背景的可分辨程度、噪声是否同时被增强，以及不同增强设置的差异。

### Task 2：均值滤波与不同滤波器尺寸比较

必须完成：

1. 自行实现并应用一个 **5 × 5 averaging filter（均值滤波器）**；
2. 继续尝试不同尺寸的滤波器；
3. 比较各 image smoothing 方法的结果；
4. 对不同平滑结果作出评论。

需要体现“不同 filter size 的实验与比较”，不能只展示 5 × 5 的单一结果。比较时应同时考虑去噪能力和字符边缘/细节损失。

### Task 3：提取包含指定字符行的子图像

必须创建一个 sub-image（子图像），该子图像需要包含以下整行字符：

```text
HD44780A00
```

应在报告中清楚展示提取范围和提取后的图像，以证明目标字符行完整保留。

### Task 4：将 Task 3 子图像二值化

必须把 Task 3 得到的 sub-image 转换为 binary image（二值图像）。

报告需要展示该阶段的中间结果。若尝试多种 thresholding 或预处理方案，应说明最终选择及理由。

### Task 5：确定字符轮廓

必须确定图像中字符的 outline(s)（轮廓）。

由于原文使用复数形式 `outline(s)`，实现应能对图像中的字符轮廓进行检测和展示，而不只是口头描述。

### Task 6：字符分割与标记

必须对图像进行 segmentation，使不同字符：

1. 尽可能清楚地彼此分离；
2. 被分别标记（label）。

报告中应展示字符分割和 labeling 的可视化中间结果，并讨论粘连字符、断裂字符、噪声区域或误分割等问题。

### Task 7：字符分类与两类方法比较

CANVAS 提供数据文件：

```text
dataset_2026.zip
```

#### 3.7.1 数据集划分

必须把数据集分为两部分：

- **75%**：training set；
- **剩余 25%**：test set。

原文要求使用这两个子数据集完成 Task 7 的三个子任务。PDF 没有进一步规定是否需要 validation set，也没有指定随机种子、分层抽样方式或交叉验证方法。

#### Subtask 7-1：CNN 分类器

必须设计一个 CNN（Convolutional Neural Network），用于分类 Figure 1 中显示的每个字符。

需要在报告中说明 CNN 的结构和处理流程，并报告实验结果。

#### Subtask 7-2：non-CNN 分类系统

必须设计一个不使用 CNN 的分类系统。具体要求为：

1. 使用 non-CNN-based method，或多种此类方法的组合；
2. 方法必须从课程 Part 2 已讲授的方法中选择；
3. 目标同样是分类 Figure 1 中显示的每个字符。

因此，不能只比较两个不同 CNN；第二套方法必须属于课程 Part 2 范围内的 non-CNN 方法或方法组合。

#### Subtask 7-3：结果报告、比较和解释

必须完成：

1. 报告 Subtask 7-1 的 CNN 结果；
2. 报告 Subtask 7-2 的 non-CNN 结果；
3. 比较两种方法的 effectiveness（有效性/效果）；
4. 比较两种方法的 efficiency（效率）；
5. 对两种方法结果之间的差异给出小组自己的解释。

这里的要求不只是列出结果。报告必须包含“比较”和“原因解释”，并且解释应来自小组自己的分析与推理。

#### Task 7 的额外实验要求

在完成 Task 7 时，还必须进行以下实验：

1. data preprocessing，例如对输入图像进行 padding 或 resizing；
2. hyperparameter tuning（超参数调优）；
3. 在报告中讨论实验发现；
4. 分析所提出方法对上述变化的 sensitivity（敏感程度）。

#### Task 7 的明确禁令

> **不得使用 Figure 1 作为分类器的 training data。**

也就是说，Figure 1/`charact2.bmp` 是最终需要处理和分类的目标图像，不能把其中的字符裁出后混入训练集。训练数据应来自 `dataset_2026.zip` 按规定划分出的 training set。

## 4. 小组组织与代码要求（Organization）

### 4.1 小组与评分方式

- 全班按小组完成项目；
- 每组会被分配一个 group number；
- 每组必须提交 project report 和对应 implementation code；
- 项目根据这些提交物进行评估；
- 同组所有成员获得相同分数。

### 4.2 实现语言

代码必须使用以下一种语言开发：

- MATLAB；或
- Python。

### 4.3 可使用的软件与 toolbox

PDF 明确允许使用：

- MATLAB Image Processing Toolbox；
- 与 machine learning 相关的其他 MATLAB toolboxes。

同时，PDF 强烈鼓励学生自行开发和实现代码，避免过度依赖 readily available software packages。这里的“自行实现”属于强烈建议；允许使用 toolbox 并不等于可以不解释核心算法和流程。

### 4.4 外部资源引用

以下内容只要被使用，都必须在报告中 properly acknowledged and referenced（适当致谢并引用）：

- external sources；
- libraries；
- toolboxes；
- third-party code。

这项要求出现在 PDF 脚注中，但仍是明确要求，不能遗漏。

## 5. 报告必须包含的内容（Report Submission）

报告至少必须包括以下内容：

1. **Title page**：列出所有小组成员的姓名和 student number；
2. **Introduction**：介绍项目问题；
3. **Algorithms and flow charts**：说明所用算法，并提供对应流程图；
4. **Intermediate results**：提供每个 image-processing stage 的中间结果截图；
5. **Discussion and conclusion**：说明：
   - 为什么选择当前方法；
   - 进行了哪些 investigations；
   - 从项目中学到了什么（lessons learned）。

注意：原 PDF 在枚举时写成 `(i), (ii), (iii), (vi), (v)`，其中 `(vi)` 很可能是编号笔误。无论编号如何，`screen captures of the intermediate results at each image-processing stage` 仍是必须包含的独立内容。

此外，结合 Task 1、Task 2 和 Task 7 的明确措辞，报告还必须呈现并讨论：

- contrast enhancement 的结果；
- 不同 averaging filter sizes / smoothing results 的比较；
- CNN 与 non-CNN 方法的结果；
- 两种分类方法的 effectiveness 与 efficiency 比较；
- 两种方法结果不同的原因解释；
- preprocessing 与 hyperparameter tuning 的实验发现；
- 方法对 preprocessing 和 hyperparameter 变化的敏感性。

## 6. 报告排版要求

| 项目 | 要求 |
|---|---|
| 字体 | Arial |
| 字号 | 11 point |
| 行距 | Single-line spacing（单倍行距） |
| 最大页数 | **50 页** |
| 典型长度 | 30–40 页 |

其中“最多 50 页”是硬性上限；“典型报告 30–40 页”是参考范围，并不等同于强制最低 30 页。PDF 没有明确说明标题页、参考文献或附录是否计入 50 页，因此最稳妥的做法是让完整 PDF 不超过 50 页。

## 7. 代码运行说明要求

为了方便评分，报告必须包含一份 step-by-step guide，清楚解释如何运行 implementation code。

PDF 特别强调：grader 没有责任自行判断或摸索如何配置运行代码所需的软件环境。因此运行指南至少应让评分者可以明确知道：

- 使用 MATLAB 还是 Python；
- 从哪个入口脚本开始运行；
- 数据文件应放在哪里；
- 需要哪些 libraries、toolboxes 或依赖；
- 如何配置软件环境；
- 如何依次复现主要图像处理阶段与分类结果。

后面这些具体条目是对“step-by-step guide”和“环境不能交给 grader 猜测”的执行性展开；PDF 没有指定固定的 README 模板或依赖文件格式。

## 8. 最终打包与提交要求

### 8.1 压缩包内必须包含

- project report；
- MATLAB 或 Python code。

PDF 要求先把二者放入同一个 folder，再创建 zip file。

### 8.2 文件名

压缩包必须命名为：

```text
Group_XX.zip
```

`XX` 表示 group number，例如：

```text
Group_01.zip
```

应保留原文格式中的大写 `G`、下划线和两位小组编号。PDF 没有单独规定压缩包内部报告文件或代码文件的命名方式。

### 8.3 上传位置和时间

- 位置：CANVAS → Assignments → Project；
- 截止：**2026-11-20 23:59 Singapore time**。

PDF 没有说明迟交处罚、允许的补交时间或多次覆盖上传规则；这些内容应以 CANVAS 的实时设置和课程公告为准。

## 9. 评分依据（Evaluation）

PDF 没有给出各部分的具体百分比分值，但明确列出了四类主要评分因素。

### 9.1 要求覆盖与技术正确性

评估项目工作：

- 在多大程度上覆盖作业要求；
- 是否产出 technically sound（技术上合理）的结果；
- 结果是否 accurate（准确）。

### 9.2 方法与实现质量

评估所提出方法及实现的：

- effectiveness；
- robustness；
- overall standard（整体水准）。

### 9.3 投入、调查深度与主动性

评估小组展现出的 commitment，包括：

- scope of work（工作范围）；
- depth of investigation（调查/实验深度）；
- 克服挑战时展现的 initiative（主动性）。

### 9.4 报告清晰度与组织

评估书面报告的 clarity 和 organization，尤其包括以下内容的呈现与讨论质量：

- methodology；
- results；
- conclusions。

## 10. PDF 没有明确规定的事项

以下内容在该 PDF 中没有给出明确要求，不应擅自当作硬性规定：

- CNN 或 non-CNN 方法的指定网络结构/具体算法；
- 必须使用的评价指标；
- validation set 的比例；
- 随机种子或 train/test split 的具体抽样规则；
- 最低分类准确率；
- 各评分项的百分比；
- 报告文件的格式或文件名（PDF 只明确了 zip 包命名）；
- 参考文献格式；
- 迟交处罚；
- 标题页、参考文献、附录是否计入 50 页；
- 是否必须提交预训练模型权重；
- 是否允许使用预训练模型；
- 是否必须提交 README、`requirements.txt` 或 Conda environment file。

如 CANVAS、课堂通知或后续公告对这些事项给出补充要求，应以更新后的官方要求为准。

## 11. 建议的最终自检 Checklist

以下清单前半部分直接对应 PDF 要求；末尾少量“可复现性”条目是帮助满足运行指南要求的执行建议。

### 11.1 图像处理任务

- [ ] 已显示 `charact2.bmp` 原始图像。
- [ ] 已进行 contrast enhancement 实验并评论结果。
- [ ] 已实现并应用 5 × 5 averaging filter。
- [ ] 已测试其他 filter sizes，并比较/评论平滑结果。
- [ ] 已提取完整包含 `HD44780A00` 的 sub-image。
- [ ] 已把该 sub-image 转换为 binary image。
- [ ] 已确定并展示字符 outline(s)。
- [ ] 已将不同字符尽可能清楚地分割并 label。
- [ ] 已保存每个 image-processing stage 的中间结果截图。

### 11.2 分类任务

- [ ] 已将 `dataset_2026.zip` 数据划分为 75% training / 25% test。
- [ ] 已设计并实现 CNN 分类器。
- [ ] 已从课程 Part 2 方法中选择并实现 non-CNN 分类系统。
- [ ] 两种方法均用于分类 Figure 1 中的字符。
- [ ] 未将 Figure 1 / `charact2.bmp` 用作训练数据。
- [ ] 已实验 data preprocessing，例如 padding/resizing。
- [ ] 已进行 hyperparameter tuning。
- [ ] 已报告 CNN 和 non-CNN 的结果。
- [ ] 已比较两者的 effectiveness。
- [ ] 已比较两者的 efficiency。
- [ ] 已由小组自己解释两种方法结果差异。
- [ ] 已讨论方法对 preprocessing 和 hyperparameter 变化的 sensitivity。

### 11.3 报告

- [ ] 标题页包含全部成员姓名和 student number。
- [ ] 包含 problem introduction。
- [ ] 描述全部主要 algorithms。
- [ ] 为算法提供 flow charts。
- [ ] 包含各 image-processing stage 的中间结果截图。
- [ ] 讨论 chosen method 的 rationale。
- [ ] 说明进行了哪些 investigations。
- [ ] 总结 lessons learned。
- [ ] 包含 discussion 和 conclusion。
- [ ] 包含可逐步执行的代码运行指南。
- [ ] 所有 external sources、libraries、toolboxes 和 third-party code 均已致谢和引用。
- [ ] 使用 11-point Arial、single-line spacing。
- [ ] 完整报告不超过 50 页。
- [ ] 报告中的 ideas、analysis 和 reasoning 均来自项目小组，LLM 仅用于允许的 language editing。

### 11.4 代码与提交

- [ ] 实现代码使用 MATLAB 或 Python。
- [ ] 评分者无需猜测即可配置环境并运行代码。
- [ ] 建议在干净环境中按报告指南完整复现一次结果。
- [ ] 报告和全部必要代码已放入同一提交文件夹。
- [ ] 建议确认程序使用相对路径，且压缩包中包含运行所需的自有文件。
- [ ] zip 包名严格为 `Group_XX.zip`，`XX` 与分配的小组编号一致。
- [ ] 最终 zip 可正常解压，报告可打开，主要代码入口可运行。
- [ ] 已在 **2026-11-20 23:59 Singapore time** 前上传至 CANVAS → Assignments → Project。
