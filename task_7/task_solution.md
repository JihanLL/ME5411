# Task 7 implementation and reproducibility notes

`task_7.py` runs both required classifiers from the repository root:

```bash
python task_7/task_7.py --data dataset_2026 --output task_7/outputs
```

Create the supplied Conda environment first with
`conda env create -f task_7/environment.yml`, then activate it using
`conda activate cvtask`.

The implementation uses PyTorch, NumPy, Pillow and Matplotlib. It automatically
uses CUDA when available; `--workers 0` is the deterministic default. A short
smoke check is available with `--smoke --output task_7/outputs_smoke`, so the
reported experiment artifacts are preserved.

The script hashes decoded grayscale pixels and keeps every exact duplicate in
one partition. A stratified outer 75/25 split is produced, followed by a
grouped validation split from the 75% training portion for hyperparameter
selection. Candidate input sizes are 32 and 48 pixels, with learning rates
`1e-3` and `3e-4`. Pixel mean and standard deviation are calculated from the
training partition and reused for validation/test; no test statistics are used.

The CNN has convolutional blocks with BatchNorm, ReLU, max pooling and global
average pooling. The non-CNN method is an MLP regression model: flattened
pixels pass through Linear, BatchNorm, GELU and Dropout layers, and the seven
outputs are trained against one-hot targets with mean squared error. The class
is the largest output; the final reported MLP uses three independently seeded
models whose scores are averaged.

Important output files are:

* `task7_metrics.json`: complete configuration, split audit, selected settings,
  held-out metrics, confusion matrices and optional Figure 1 predictions;
* `method_comparison.csv`: compact CNN/MLP effectiveness and efficiency table;
* `tuning_results.csv`, `tuning_results.json`, `tuning_curves.png`: validation
  hyperparameter and preprocessing comparison;
* `split_summary.json`, `split_manifest.csv`: class counts and duplicate-leakage
  audit;
* `cnn_confusion_matrix.png`, `mlp_confusion_matrix.png`: test confusion plots;
* `cnn_*px_seed*.pt`, `mlp_*px_seed*.pt`: checkpoints with normalization
  statistics and class names.
* `target_predictions.json`: predictions and score vectors for the ten
  segmented characters from the supplied Figure 1 image.

When Task 3 and Task 6 outputs are present, the script also applies each final
model to the ten segmented crops from `Test_Image_2026.bmp` and records the
predicted string under `figure1_inference` in `task7_metrics.json`. These
predictions are kept separate from the dataset test score because Figure 1 is
never included in training or hyperparameter selection.

## Selection and training configuration

Every grid candidate is **one network**, including the MLP. The eight
configurations are two network kinds × sides `{32,48}` × initial learning
rates `{1e-3,3e-4}`. Each is run once with seed
`20261013 + 100*side + int(100000*lr) + offset`, where offset is 0 for CNN and
1 for MLP. Consequently, initialization and batch order also vary across
candidates; small score differences are exploratory rather than isolated
causal effects of a hyperparameter.

Configuration selection maximizes validation accuracy, with ties resolved by
smaller input size, then smaller learning rate. Three MLP configurations tie
at 100% validation accuracy. The selected 32-pixel / `3e-4` configuration
follows this rule; it is not uniquely most accurate. Within a candidate,
validation loss breaks an accuracy tie when selecting the best epoch.

Fitting batch size is 128; evaluation batch size is 256. AdamW uses default
betas `(0.9,0.999)` and epsilon `1e-8`, with weight decay `1e-4`. Tuning runs
at most 18 epochs, with early-stop patience 5 and ReduceLROnPlateau
`mode=max, factor=0.5, patience=2, min_lr=initial_lr/20`. Final training uses
all 5334 outer-training examples and CosineAnnealingLR with `T_max` equal to
the selected best epoch: 15 CNN epochs and 17 epochs per MLP member.
Final seeds are CNN 20268013 and MLP 20268014, 20268051, 20268088.
No separate `--epochs-final` option is used: the final budget comes from
validation selection.

## Figure 1 preprocessing repair and inference-only reproduction

```bash
python task_7/task_7.py --data dataset_2026 --output task_7/outputs --inference-only
```

This command reloads the saved selected checkpoints and the original split
manifest. It preserves the controlled held-out metrics and training times.
It computes class-agnostic median glyph bounding-box width and height from
**tuning-train only** (4802 images): 0.609375 and 0.7578125 relative to source
canvas width/height. Dark pixels `<128` define dataset foreground.

The fixed new target preparation uses the binary crops already produced by
Task 6. It keeps the largest 8-connected foreground component, tight-crops
its box, inverts to black on white, independently rescales width and height
to the train-derived medians, and centers the result on a white canvas.
The original checkpoint normalization then applies. This single rule was
fixed before inspecting its predictions; target labels, target model scores,
and held-out test measurements do not select its parameters. Aspect ratio
can change because width and height are aligned separately.

Both paths and all scores are saved in `target_predictions.json` and
`task7_metrics.json`; the original grayscale baseline remains under
`baseline_gray_inference`. Checkpoint names and SHA-256 digests document that
network parameters were unchanged. `target_preprocessing.png` visualizes the
original and aligned inputs.

| Inference path | CNN prediction | CNN row accuracy | MLP ensemble prediction | MLP row accuracy |
| --- | --- | --- | --- | --- |
| Original inverted grayscale + square padding | `AAAAAAAAAA` | 1/10 (10%) | `HD447H0A00` | 9/10 (90%) |
| Fixed binary / tuning-train geometry alignment | `AD44780A80` | 8/10 (80%) | `HD44780A00` | 10/10 (100%) |

The known reference is `HD44780A00`. Accuracy is assessed **after inference**;
this is one ten-character target row, not a new held-out benchmark. The
clean-dataset test results remain CNN 99.3251% and MLP ensemble 98.7064%.
Further domain-aware retraining would need additional allowed training
images or training-set-derived augmentations; Figure 1 and the held-out test
partition must remain outside fitting and model selection.
