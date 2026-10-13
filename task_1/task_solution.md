# Task 1 问题解决思路：原图显示与对比度增强

## 1. 任务要求

Task 1 的原始要求是：

1. 在屏幕上显示原始图像；
2. 对图像进行 contrast enhancement（对比度增强）实验；
3. 对实验结果进行评论和分析。

因此，本任务不能只给出一张“增强后的图片”。至少应展示原图、尝试若干种增强方法或参数，并比较它们对字符可见性、背景噪声、局部细节和后续处理的影响。

## 2. 输入图像的初步观察

当前项目中的目标图像为：

```text
dataset_2026/Test_Image_2026.bmp
```

已检查到的基本属性如下：

| 属性 | 数值 |
|---|---|
| 图像尺寸 | 990 × 367 pixels |
| 图像模式 | RGB |
| 数据类型 | `uint8` |
| 像素值范围 | 41–255 |
| RGB 平均值 | 约 `[89.90, 91.48, 96.75]` |

从视觉上可以观察到：

- 字符比背景更亮，但字符与背景的灰度差异并不稳定；
- 图像有轻微颜色偏差，不过识别信息主要来自亮度而不是颜色；
- 背景存在明显、密集的环状纹理和亮点噪声；
- 字符本身也由不均匀的亮斑组成，边缘模糊且局部断裂；
- 直接进行强烈的全局增强可能同时放大背景噪声。

因此，Task 1 的重点不只是“让图更亮”，而是提高字符和背景之间的可分辨性，同时尽量避免把背景纹理增强到与字符同样显眼。

> 文件名说明：项目 PDF 将 Figure 1 的图像称为 `charact2.bmp`，当前目录中实际存在的是 `dataset_2026/Test_Image_2026.bmp`。实现时应以老师最终指定的测试图像为准，并在代码中集中设置输入路径，避免把文件名写死在多个位置。

## 3. 可使用的开发工具

### 3.1 推荐：Python

README 指定项目使用 Conda 环境：

```bash
conda activate cvtask
```

当前 `cvtask` 环境已经具备：

| 工具 | 可用于 |
|---|---|
| NumPy | 像素矩阵运算、灰度转换、直方图、contrast stretching、gamma correction |
| Matplotlib | 显示原图/增强图、绘制直方图、保存报告图片 |
| Pillow（PIL） | 读取 BMP、转换 RGB/灰度图、保存结果 |
| SciPy | 后续数值处理或图像滤波实验 |
| Pandas / Seaborn | 可选，用于整理实验参数和绘制比较图 |

当前环境中没有安装：

- OpenCV（`cv2`）；
- scikit-image（`skimage`）。

这两个包不是完成 Task 1 的必要条件。基础方法完全可以用 NumPy、Pillow 和 Matplotlib 自行实现，这也更容易在报告中解释算法。如果后续决定使用 CLAHE、Otsu thresholding、connected components 等成熟实现，可以安装 OpenCV 或 scikit-image，但必须：

1. 在运行说明中记录安装方法和版本；
2. 在报告中正确 acknowledge/reference 使用的 library；
3. 仍然解释方法原理、参数和结果，不能只展示库函数调用。

### 3.2 可选：MATLAB

如果小组选择 MATLAB，可使用 Image Processing Toolbox 中的：

| MATLAB 工具 | 作用 |
|---|---|
| `imread`, `imshow` | 读取并显示图像 |
| `rgb2gray` | RGB 转灰度图 |
| `imhist` | 绘制灰度直方图 |
| `imadjust`, `stretchlim` | 线性或 percentile contrast stretching |
| `histeq` | 全局 histogram equalization |
| `adapthisteq` | CLAHE / adaptive histogram equalization |

整个项目应尽量统一使用一种语言。若 Task 1 使用 MATLAB、其他任务使用 Python，会增加环境配置、结果传递和最终复现的成本。

### 3.3 辅助工具

- Jupyter Notebook：适合快速调整参数和并排观察结果；
- 普通 Python script：更适合作为最终可复现的提交代码；
- VS Code：编辑代码、Markdown 和 LaTeX 报告；
- LaTeX：将实验流程、公式、图像和讨论整理到 `task_1/task_1.tex`；
- Git：记录小组代码和参数变化，但不要把大量临时输出或 Conda 环境本体提交进去。

推荐先用 Notebook 探索参数，确定方法后再整理为可一次运行的 Python script。

## 4. 预处理与正确显示

### 4.1 保留原图

读取图像后应立即保留一份原始数组，所有增强方法都从同一个原始输入生成，不要连续覆盖后再比较，否则无法判断某一步的真实作用。

```python
from PIL import Image
import numpy as np

rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
```

### 4.2 原图显示要求

报告中至少应显示：

- 原始 RGB 图像；
- 图像尺寸和数据类型；
- 灰度直方图，帮助说明为什么需要增强。

比较不同方法时，所有灰度图应使用相同显示范围，例如：

```python
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
```

如果让 Matplotlib 对每张图分别自动拉伸显示范围，即使像素没有真正改善，不同图也可能看起来都“对比度很好”，从而导致不公平比较。

### 4.3 RGB 转灰度

Task 1 可以先显示 RGB 原图，再将其转换为 luminance/grayscale 进行增强。常用转换为：

$$
Y = 0.299R + 0.587G + 0.114B.
$$

建议使用浮点数完成中间计算，最后显示或保存时再裁剪到 $[0,255]$ 并转换为 `uint8`，避免计算过程中的溢出和量化误差。

保留颜色的增强也可以作为补充实验，例如只增强 luminance channel，再与原色度合并。但该项目的目标是字符识别，颜色信息较弱，灰度主线通常更简单、可解释，也更容易衔接 Task 2–6。

## 5. 可尝试的对比度增强方法

建议至少选择三类方法进行实验：一种简单 baseline、一种较稳健的全局方法，以及一种局部方法。下面列出的全部方法不必都放入最终方案，但可以先实验后根据结果筛选。

### 5.1 方法 A：Min-max linear contrast stretching

把输入灰度的最小值和最大值映射到 0 和 255：

$$
I_{out}=255\frac{I-I_{min}}{I_{max}-I_{min}}.
$$

优点：

- 实现简单，是很好的 baseline；
- 保持像素的灰度顺序；
- 不需要额外 library。

缺点：

- 对极少量过暗或过亮的 outliers 很敏感；
- 当前图像已经有像素达到 255，单纯 min-max stretching 可能改善有限；
- 无法适应图像不同区域的局部亮度变化。

### 5.2 方法 B：Percentile-based contrast stretching

不直接使用绝对最小值和最大值，而是选择低、高百分位数 $p_l$ 和 $p_h$：

$$
I_{out}=255\frac{I-p_l}{p_h-p_l},
$$

然后把结果裁剪到 $[0,255]$。可尝试：

- 1st–99th percentile；
- 2nd–98th percentile；
- 5th–95th percentile。

优点：

- 忽略少量异常亮点和暗点，比 min-max 更稳健；
- 对当前含有亮点噪声的图像比较合适；
- 只需 NumPy 即可自行实现。

缺点：

- 过于激进的百分位数会造成 highlight/shadow clipping；
- 参数需要通过实验确定。

这是 Task 1 推荐作为主方法的候选之一。

### 5.3 方法 C：Gamma correction

先把灰度归一化到 $[0,1]$，再进行幂律变换：

$$
I_{out}=I_{in}^{\gamma}.
$$

- 当 $\gamma<1$ 时，中低灰度区域会变亮；
- 当 $\gamma>1$ 时，中低灰度区域会变暗；
- 当 $\gamma=1$ 时，图像不变。

建议尝试：

```text
gamma = 0.5, 0.8, 1.0, 1.2, 1.5
```

优点：

- 计算简单；
- 可以有针对性地调整暗区和中间灰度；
- 适合作为参数敏感性实验。

缺点：

- 它改变整体亮度分布，不一定能真正分开字符与背景；
- 过小的 $\gamma$ 可能使背景和噪声一起变亮；
- 过大的 $\gamma$ 可能让较弱的字符笔画消失。

### 5.4 方法 D：Global histogram equalization

Histogram equalization 使用灰度累计分布函数（CDF）重新映射像素，使可用灰度范围得到更充分利用。

优点：

- 自动增强全局对比度；
- 不需要手动选输入上下限；
- 可直接比较增强前后的 histogram。

缺点：

- 容易显著放大当前图像中的背景纹理和环状噪声；
- 可能产生不自然的亮度分布；
- 全局映射无法针对局部照明变化。

该方法值得作为对照组，即使结果不是最好，也能在报告中解释为什么“更大的全局对比度”不等于“更好的字符可识别性”。

### 5.5 方法 E：CLAHE

CLAHE（Contrast Limited Adaptive Histogram Equalization）将图像划分为局部 tiles，在每个区域进行直方图均衡，同时用 clip limit 限制噪声放大，再对相邻 tiles 插值。

可实验参数包括：

- tile/grid size，例如 8 × 8、16 × 16；
- clip limit，例如 1、2、4；
- 不同局部区域大小与噪声放大的关系。

优点：

- 能处理局部对比度不均；
- 比普通 adaptive histogram equalization 更能抑制噪声放大；
- 通常适合字符亮度在空间上不均匀的图像。

缺点：

- 参数较多；
- clip limit 太高仍会强化环状背景；
- tile 太小可能产生局部伪影，太大又接近全局均衡；
- 当前 `cvtask` 环境没有 OpenCV/scikit-image，若不自行实现则需要新增依赖。

CLAHE 适合作为高级候选方法，但应与简单的 percentile stretching 比较，确认增加的复杂度是否真的改善了字符可见性。

### 5.6 方法 F：局部背景校正后再拉伸

如果发现图像存在明显的低频照明不均，可以先估计背景 $B(x,y)$，再进行：

$$
I_{corrected}(x,y)=I(x,y)-B(x,y),
$$

随后重新归一化或进行 percentile stretching。背景可由大尺度 Gaussian filter、median filter 或 morphological opening 估计。

优点：

- 能减少缓慢变化的背景亮度；
- 可能使不同位置的字符具有更一致的对比度。

缺点：

- filter size 选择不当会把字符本身当成背景；
- 会与 Task 2 的 smoothing 内容发生部分重叠；
- 当前主要干扰是密集局部纹理，背景校正未必足够。

因此，它可以作为补充实验，不建议在没有证据时直接作为 Task 1 的唯一方案。

## 6. 方法比较与评价指标

### 6.1 视觉评价

每种方法都应在相同显示尺度下比较以下问题：

1. `T12` 和 `HD44780A00` 的笔画是否更清楚；
2. 较弱、断续的笔画是否得到保留；
3. 背景环状纹理是否被过度增强；
4. 字符边缘是否出现 clipping、halo 或局部伪影；
5. 不同位置的字符是否都得到改善；
6. 结果是否有利于后续二值化、轮廓提取和字符分割。

### 6.2 定量评价

Task 1 没有指定必须使用的评价指标，但为了使评论更有依据，可以记录：

| 指标 | 用途 | 注意事项 |
|---|---|---|
| 灰度 histogram | 观察动态范围和分布变化 | 分布更宽不一定代表字符更清楚 |
| Mean / standard deviation | 描述整体亮度和全局对比度 | 容易被背景噪声影响 |
| 1%、50%、99% quantiles | 检查暗区、中间值和亮区变化 | 可判断 stretching 是否过强 |
| Saturation ratio | 统计变成 0 或 255 的像素比例 | 比例过高通常说明 clipping 严重 |
| Michelson contrast | 比较选定亮/暗区域 | 适合相对均匀的局部 ROI |
| CNR | 比较字符 ROI 与背景 ROI 的可分性 | ROI 必须固定，避免人为偏向某一方法 |

若选取固定的字符区域和邻近背景区域，可计算 contrast-to-noise ratio：

$$
\mathrm{CNR}=\frac{|\mu_c-\mu_b|}{\sqrt{\sigma_c^2+\sigma_b^2}},
$$

其中 $\mu_c,\sigma_c$ 是字符区域的均值和标准差，$\mu_b,\sigma_b$ 是背景区域的均值和标准差。CNR 越大，通常表示字符与背景越容易区分。

ROI 的位置必须对所有方法保持完全一致，并在报告中展示，否则比较没有说服力。由于当前字符和背景纹理重叠较强，定量指标应与视觉结果一起解释，不能只根据单个数字下结论。

### 6.3 面向后续任务的评价

本项目后续需要 binary conversion、outline detection 和 character segmentation。因此最终增强方法还应通过一个简单的 downstream test 检查：

- 在相同或合理阈值下，字符是否更容易从背景分离；
- 是否产生大量与字符无关的前景噪声；
- 是否使字符断裂或互相粘连。

这部分可以作为选择最终方法的依据，但 Task 1 中不需要提前完成 Task 4–6 的全部实现。

## 7. 推荐实验流程

### Step 1：读取并记录输入

- 读取 `Test_Image_2026.bmp`；
- 记录 shape、dtype、min/max、mean 和 quantiles；
- 显示 RGB 原图；
- 保存原始灰度 histogram。

### Step 2：建立灰度 baseline

- 使用固定公式转换为 grayscale；
- 固定 `vmin=0, vmax=255` 显示；
- 不进行增强的灰度图作为 baseline。

### Step 3：实现简单全局方法

- min-max stretching；
- percentile stretching：1–99、2–98、5–95；
- gamma correction：$\gamma=0.5,0.8,1.0,1.2,1.5$。

### Step 4：实验 histogram-based 方法

- global histogram equalization；
- 如果加入 OpenCV/scikit-image，则实验不同参数的 CLAHE。

### Step 5：统一比较

- 使用同一版式并排显示原图和各增强结果；
- 所有图使用统一的显示尺度；
- 绘制相应 histogram；
- 记录 quantiles、standard deviation、saturation ratio；
- 使用固定 ROI 计算 CNR；
- 观察背景纹理是否被放大。

### Step 6：选定主方法

建议优先从以下两者中选择最终主方法：

1. **Percentile stretching**：实现简单、可解释、无新增依赖，对异常亮点更稳健；
2. **低 clip limit 的 CLAHE**：若实验能证明它对局部字符对比度有明显改善，且没有过度放大背景。

Gamma correction 可与主方法组合，例如先做稳健拉伸，再进行温和 gamma 调整。但组合步骤越多，越需要用消融比较证明每一步确有作用。

### Step 7：保存可复现结果

建议把每次实验的以下信息保存在代码或结果表中：

- 方法名称；
- 全部参数；
- 输入文件；
- 输出文件；
- 定量指标；
- 简短观察结论。

输出图像建议保存为 PNG，避免 JPEG 压缩引入新的伪影。文件名应包含方法和参数，例如：

```text
task1_original.png
task1_gray.png
task1_stretch_p02_p98.png
task1_gamma_080.png
task1_histeq.png
task1_clahe_clip02_grid08.png
task1_comparison.png
task1_histograms.png
```

## 8. 推荐的代码结构

可在 `task_1/` 下采用如下结构：

```text
task_1/
├── task_1.py              # 最终可复现脚本
├── task_solution.md       # 本文：问题解决思路
├── task_1.tex             # Task 1 报告内容
└── outputs/               # 生成的图片和实验结果
```

`task_1.py` 可以拆分为以下函数：

```text
load_image(path)
rgb_to_gray(rgb)
linear_stretch(gray)
percentile_stretch(gray, low_percentile, high_percentile)
gamma_correct(gray, gamma)
histogram_equalize(gray)
compute_metrics(gray, character_roi, background_roi)
plot_comparison(results)
save_results(results, output_dir)
```

这样可以保证所有方法使用相同输入、评价函数和绘图方式，也便于后续在 Task 2–6 中复用灰度图和增强结果。

## 9. 报告中应该呈现的内容

`task_1/task_1.tex` 最终建议包含：

1. Task 1 的目标；
2. 原图属性和主要困难；
3. RGB-to-grayscale 方法；
4. 每种增强方法的简要原理和公式；
5. 参数设置与选择理由；
6. 原图、灰度图和增强图的统一尺度对比；
7. 原始与增强后的 histogram；
8. 一张定量结果表；
9. 对字符可见性、噪声放大、clipping 和局部伪影的讨论；
10. 最终方法选择，以及该结果如何支持后续 Task 2–6。

评论不能只写“效果更好”或“对比度增加”。更有说服力的表达应说明具体变化，例如：

> 2nd–98th percentile stretching 提高了暗背景与较弱字符笔画之间的视觉差异，同时比 global histogram equalization 引入更少的背景亮点。后者虽然增加了全局灰度标准差，但也显著强化了环状纹理，因此不适合作为后续二值化的唯一输入。

上面的句子只是写作结构示例，最终结论必须以小组实际运行得到的图像和指标为依据。

## 10. 风险与注意事项

1. **不要把 brightness adjustment 当成 contrast enhancement。** 整体加一个常数只会改变亮度，不一定增加字符与背景的差异。
2. **不要依赖自动显示范围。** 必须固定显示尺度，才能公平比较。
3. **不要只看 histogram。** 背景噪声被放大也可能让 histogram 更分散。
4. **避免过度 clipping。** 大量像素变为 0 或 255 会损失字符细节。
5. **避免只展示最佳结果。** 至少保留有代表性的 baseline 和失败/较差方法，才能形成有效讨论。
6. **参数必须记录。** 不能只写“使用 gamma correction”而不写 $\gamma$。
7. **保证下游一致性。** Task 1 选定的增强结果应能清楚传递给 Task 2 或后续流程。
8. **区分强制要求与研究扩展。** 多种指标、CLAHE 和背景校正是提升分析质量的方案，不是 PDF 明文指定的方法。
9. **引用第三方工具。** 如果使用 OpenCV、scikit-image、Pillow 或外部代码，应按项目要求在最终报告中 acknowledge/reference。
10. **保留小组自己的分析。** 根据课程 LLM policy，方法选择、实验、结果分析和推理必须由项目小组完成。

## 11. Task 1 完成标准 Checklist

- [ ] 能从统一配置的路径读取正确 BMP 图像。
- [ ] 显示并保存原始 RGB 图像。
- [ ] 记录图像尺寸、类型、像素范围和 histogram。
- [ ] 生成并显示 grayscale baseline。
- [ ] 至少实现一种基础 contrast enhancement 方法。
- [ ] 使用多种方法或多组参数进行实验。
- [ ] 所有比较图使用相同显示范围和版式。
- [ ] 比较字符清晰度、背景噪声、clipping 和伪影。
- [ ] 使用视觉证据，并尽可能提供定量指标。
- [ ] 明确说明最终选择的方法及原因。
- [ ] 保存中间结果，供最终报告截图使用。
- [ ] 代码可在 `cvtask` 环境中从头运行。
- [ ] 新增依赖和第三方工具已记录并准备在报告中引用。
- [ ] 结论来自小组实际实验，不把示例性判断当作最终结果。
