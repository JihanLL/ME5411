# ME5411 image processing and character recognition

The implementation uses Python. Run the commands below from the project root
(the directory containing `main.tex`, `common.py`, and `task_1/` through
`task_7/`). Each task contains its entry-point script, report fragment,
implementation notes, and generated results. `main.pdf` is the report.

## 1. Place the supplied dataset

Keep this layout and these names:

```text
dataset_2026/
├── Test_Image_2026.bmp
├── 0/   # 1,016 PNG images
├── 4/   # 1,016 PNG images
├── 7/   # 1,016 PNG images
├── 8/   # 1,016 PNG images
├── A/   # 1,016 PNG images
├── D/   # 1,016 PNG images
└── H/   # 1,016 PNG images
```

The dataset contains 7,112 labelled, 128 × 128 grayscale images. The chip BMP
and its segmented character crops are used only for the final target-image
inference. Do not move them into a labelled class directory.

## 2. Create and activate the Python environment

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) if
Conda is not already available, then run in a Bash shell:

```bash
conda env create -f task_7/environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cvtask
python -c "import sys, torch, numpy, PIL, matplotlib; print(sys.version); print(torch.__version__, numpy.__version__, PIL.__version__, matplotlib.__version__); print('CUDA:', torch.cuda.is_available())"
```

If `cvtask` already exists, update it with
`conda env update -n cvtask -f task_7/environment.yml` instead of the creation
command, then activate it and check the versions. The supplied environment file pins Python 3.8,
PyTorch 2.0.0, torchvision 0.15.1, CUDA runtime 11.8, NumPy 1.24.2,
Pillow 9.4.0, and Matplotlib 3.7.1. The measured experiments used Ubuntu 20.04,
Python 3.8.10, PyTorch 2.0.0+cu118, and an NVIDIA GeForce RTX 4080 SUPER.

Tasks 1–6 use NumPy, Pillow, and Matplotlib on the CPU. Task 7 automatically
uses CUDA when available and otherwise runs on the CPU. The CUDA-enabled
environment can also execute on the CPU; to force that mode, prefix the Task 7
command with `CUDA_VISIBLE_DEVICES=""`. A working NVIDIA driver is needed only
for GPU execution. Reported training and inference times depend on the device.
All plots use the noninteractive Agg backend, so a desktop display is not
required.

## 3. Reproduce Tasks 1–6 in order

```bash
python task_1/task_1.py
python task_2/task_2.py
python task_3/task_3.py
python task_4/task_4.py
python task_5/task_5.py
python task_6/task_6.py
```

These scripts use fixed input and output paths derived from their own location.
They create `task_N/outputs/` and replace matching generated files there.

| Task | Input needed | Main output and downstream use |
|---|---|---|
| 1 | `dataset_2026/Test_Image_2026.bmp` | `task_1/outputs/task1_percentile_stretch_p02_p98.png`, used by Tasks 2 and 3; enhancement plots and metrics |
| 2 | Task 1 enhanced image | `task_2/outputs/task2_box_5x5.png`; comparison of five averaging-filter sizes; this is a comparison branch, not the input to Task 3 |
| 3 | Original BMP and Task 1 enhanced image | `task_3/outputs/task3_subimage.png`, used by Tasks 4–6; extraction coordinates in JSON |
| 4 | Task 3 grayscale subimage | `task_4/outputs/task4_binary.png` and `task4_binary_otsu.png`, used by Tasks 5 and 6 |
| 5 | Task 4 selected binary image and Task 3 subimage | `task_5/outputs/task5_outlines.png`, outline mask, trace example, and JSON contour records |
| 6 | Task 3 subimage and Task 4 binary images | `task_6/outputs/task6_components.csv`, labelled figures, and ten pairs of `chars/char_*_gray.png` / `char_*_bin.png` crops used by Task 7 |

Run Task 6 before Task 7 so that the Figure 1 inference has all ten character
crops. The expected row is `HD44780A00`.

## 4. Reproduce Task 7

Run the complete experiment in a separate output directory to preserve the
artifacts used by the submitted report:

```bash
python task_7/task_7.py --data dataset_2026 \
  --output task_7/outputs_reproduction
```

This runs the full two-resolution/two-learning-rate tuning grid, selects each
network on the training-pool validation subset, refits on the complete outer
75% training pool, evaluates the 25% held-out test partition, and infers the
Figure 1 characters. The default seed is `20261013`, batch size is 128,
tuning limit is 18 epochs, early-stopping patience is 5, and data-loader
workers are 0. Final training uses the selected validation epoch rather than
a separately chosen epoch limit. The complete run uses one CNN and three MLP
regression members whose scores are averaged. Different hardware or library
versions can change numerical results and timing.

For a short installation and pipeline check, use a different directory:

```bash
python task_7/task_7.py --data dataset_2026 \
  --smoke --output task_7/outputs_smoke
```

Smoke mode uses only 32 × 32 inputs, learning rate `1e-3`, at most two tuning
epochs, and one MLP member. It is not the experiment reported in `main.pdf`.
Always give the smoke output directory explicitly: the default `--output`
points to the formal `task_7/outputs/` directory.

Each output directory contains:

| Artifact | Purpose |
|---|---|
| `split_summary.json`, `split_manifest.csv` | Partition counts, class labels, pixel hashes, and exact-duplicate overlap audit |
| `tuning_results.csv`, `tuning_results.json`, `tuning_curves.png` | All hyperparameter candidates and validation histories |
| `task7_metrics.json`, `method_comparison.csv` | Selected models, held-out metrics, parameter counts, and measured times |
| `cnn_confusion_matrix.png`, `mlp_confusion_matrix.png` | Held-out per-class errors |
| `cnn_*px_seed*.pt`, `mlp_*px_seed*.pt` | Trained weights, class order, and normalisation statistics |
| `target_predictions.json` | Per-method Figure 1 predictions and target preprocessing results |

The supplied formal results are in `task_7/outputs/`. The LaTeX fragments load
figures from that directory, not from `outputs_reproduction/` or
`outputs_smoke/`. To intentionally replace the formal experiment, run the full
command with `--output task_7/outputs`, then update the numerical tables and
analysis in `task_7/task_7.tex` to match the new metrics before compiling.

To reproduce target-image inference with the supplied trained models, without
retraining, copy their artifacts to an independent directory and run:

```bash
cp -a task_7/outputs task_7/outputs_inference_check
python task_7/task_7.py --data dataset_2026 --inference-only \
  --output task_7/outputs_inference_check
```

Choose a new destination name if `outputs_inference_check` already exists.
This mode reads checkpoints and the split manifest from the copied directory,
fits foreground geometry using only its training rows, and updates target
predictions there. It requires both grayscale and binary Task 6 crops and the
original labelled dataset. It preserves held-out test metrics and the original
grayscale inference baseline while recording the binary-aligned inference.

## 5. Compile the report

The report needs XeLaTeX, latexmk, standard LaTeX packages, and the actual Arial
font family (regular, bold, italic, and bold italic). On Ubuntu, the TeX
dependencies can be installed with:

```bash
sudo apt-get update
sudo apt-get install latexmk texlive-xetex texlive-latex-extra \
  texlive-latex-recommended texlive-fonts-recommended \
  curl cabextract fontconfig
```

Omit `sudo` when running as root. Install the actual Arial files with the
provided installer (it verifies the download and installs the four faces in
`~/.local/share/fonts/msttcorefonts/` by default, without root privileges),
then check the resolved font:

```bash
bash scripts/install_report_fonts.sh
fc-match Arial
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -file-line-error main.tex
```

Set `REPORT_FONT_DIR` when invoking the installer to choose another directory.
The verified runtime uses `/usr/local/share/fonts/msttcorefonts/`.
`fc-match Arial` must resolve to Arial itself; an Aerial or Liberation Sans
fallback does not meet the stated font requirement. The build produces
`main.pdf`. The project `.latexmkrc` cleans auxiliary build files after a
successful compilation while retaining the PDF. To retain the log for
diagnosis, run the same build with the additional latexmk option
`-e '$success_cmd = "";'`.
