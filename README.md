# DINOGaze2

## Cross-Domain DG: Uniform224

Branch `exp/cross-domain-224` is based directly on main. The model,
last-four-layer feature fusion and auxiliary gaze losses are unchanged.
The SAT comparison lives separately on `exp/cross-domain-topk-st`.

```bash
python cross_domain.py train --source ethxgaze --exp dg_eth_224
python cross_domain.py train --source gaze360 --exp dg_gaze360_224
python cross_domain.py test --source ethxgaze --checkpoint /path/to/source.pt --output save/dg_eth_224_eval.json
```

Each source is trained once; source validation MAE selects one checkpoint.
That same checkpoint is evaluated on all 15 MPIIFaceGaze subjects and all four
EYEDIAP clusters, without target training, BN adaptation, checkpoint selection
or eye annotations. Source test is separately evaluated by the original trainer
after source-val selection. It is not used to select weights.

`config/cross_domain.yaml` declares source YAMLs and target label groups.
This branch reads native 224x224 images directly from `/home/hqh/dataset/gaze`
for both source and target domains. No interpolation is performed; unexpected
image sizes raise an error. Source batch size remains 256 as in the main YAMLs.
Top-K-ST separately uses 448 input and batch size 200 due to GPU memory limits;
that branch is unchanged. Report this batch-size difference, and audit crop
alignment between the separately prepared 224/448 datasets when comparing.
Main's existing in-domain configs and no-resize reader default are unchanged.

`--epochs`, `--batch-size`, `--workers`, `--seed`, `--limit` and `--save-root`
are supported. `--limit` marks results smoke-only. Reusing a training experiment
directory is rejected. Target results go to `cross_domain.json`, separate from
`source_result.json`: `mae` is sample-weighted angular error; `macro_group_mae`
is the unweighted mean over 15 subjects or 4 clusters. Report the same metric
for both models. Per-group errors, source checkpoint and effective model are
recorded. Test mode requires a source-val checkpoint with metadata.

The supplied protocol uses existing GazeHub labels, not a claim of exact
equivalence to every published DG table. Audit gaze normalization, frame
subsampling, EYEDIAP session selection and Gaze360 angular filtering before
cross-paper comparisons. No extra angular filtering is applied here.

## Original Baseline

Minimal, config-driven DINOv2 gaze-estimation baseline for four GazeHub-format
datasets:

- ETH-XGaze
- MPIIFaceGaze
- EYEDIAP
- Gaze360

The project reads the supplied `.label` files directly. It does not create or
depend on intermediate JSON split files.

## Layout

```text
config/                  Dataset and protocol YAML files
model.py                 DINOv2 backbone, layer fusion, and gaze heads
reader.py                One GazeHub label reader for all datasets
train.py                 Config-driven training for fixed and CV protocols
test.py                  One-checkpoint evaluation entry point
utils.py                 Geometry, logging, scheduler, checkpoints
save/<experiment>/<fold>/
  checkpoint/            Best-k checkpoints selected by monitor MAE
  log/                   Text logs, TensorBoard, and prediction grids
```

## Setup

Install a PyTorch and torchvision build that matches the local CUDA driver, then
install the remaining packages:

```bash
pip install -r requirements.txt
```

`DINOGaze` loads DINOv2 with `torch.hub`. Set `DINO_REPO_PATH` to an existing
local DINOv2 checkout to avoid a network download:

```bash
export DINO_REPO_PATH=/home/hqh/projects/dinov2
```

The baseline was verified with Python 3.10.19, PyTorch 2.4.0+cu121,
torchvision 0.19.0+cu121, xFormers 0.0.27.post2, and DINOv2 commit
`7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c`.

The default YAML paths expect the prepared datasets at
`/home/hqh/dataset/gaze`.

## Train

```bash
cd /home/hqh/projects/DINOGaze2

python train.py --config config/ethxgaze.yaml --exp vits14_eth
python train.py --config config/gaze360.yaml --exp vits14_gaze360
python train.py --config config/mpiifacegaze.yaml --exp vits14_mpii
python train.py --config config/eyediap.yaml --exp vits14_eyediap
```

MPIIFaceGaze expands to its 15 leave-one-subject-out folds. EYEDIAP expands to
four leave-one-cluster-out folds. A quick per-fold check is useful before a full
cross-validation run:

```bash
python train.py --config config/mpiifacegaze.yaml --exp probe --fold p00 --epochs 1 --limit 512
```

ETH-XGaze and Gaze360 select checkpoints with `val.label`; their `test.label`
is evaluated only once after selection. MPIIFaceGaze and EYEDIAP have no separate
validation split in these protocols, so their held-out fold is the monitor.

Training hyperparameters live in each YAML file. The baseline trains all DINOv2
ViT-S/14 parameters and fuses blocks 8--11 with a small Transformer. Its fusion
depth, attention heads, FFN width, and dropout are configurable:

```yaml
model:
  name: dinov2_vits14
  out_indices: [8, 9, 10, 11]
  trans_dim: 128
  fusion_layers: 6
  fusion_heads: 8
  fusion_ff: 512
  fusion_dropout: 0.1
```

`trans_dim` must be divisible by `fusion_heads`. No data augmentation is
applied. AMP is enabled automatically for CUDA training and disabled on CPU.

## Baseline Results

| Dataset | Protocol | Test MAE |
| --- | --- | ---: |
| ETH-XGaze | validation-selected checkpoint | 4.287 deg |
| MPIIFaceGaze | 15-fold LOSO mean | 3.340 deg |
| EYEDIAP | 4-fold leave-one-cluster-out mean | 4.556 deg |
| Gaze360 | validation-selected checkpoint | 10.876 deg |

MPIIFaceGaze and EYEDIAP use the held-out fold for both checkpoint selection
and reporting, matching the protocol chosen for this baseline. ETH-XGaze's
reference run stopped after epoch 95/100; its best checkpoint was already
reached at epoch 68 and was evaluated on the test split afterward. Re-run the
full ETH-XGaze schedule before treating 4.287 degrees as a final publication
number. The Gaze360 reference run used 12 data-loader workers; the baseline
YAML now uses 6, which does not change the model, optimizer, data order, or
augmentation policy.

## Evaluate

The command below uses the checkpoint recorded in the fold's `result.json`,
falling back to the lowest monitor MAE in its checkpoint directory. Pass
`--checkpoint path/to/file.pt` to evaluate a specific checkpoint instead.

```bash
python test.py --config config/ethxgaze.yaml --exp vits14_eth --fold split
python test.py --config config/mpiifacegaze.yaml --exp vits14_mpii --fold p00
python test.py --config config/gaze360.yaml --exp vits14_gaze360 --fold split --split val
```

## Labels and Protocols

The model always receives and predicts `(yaw, pitch)` in radians. `2DGaze` in
MPIIFaceGaze, EYEDIAP, and Gaze360 already uses that order. ETH-XGaze's `gaze`
column is `(pitch, yaw)`, so the reader swaps it automatically. Angular MAE is
calculated from the corresponding GazeHub 3D gaze-vector convention.

`folds` can either be an explicit list (`name`, `train`, optional `val`, `test`)
or the compact `leave_one_out` declaration used by MPIIFaceGaze and EYEDIAP.
Each label entry may override `root`, `image_dir`, and `prefix`, which is enough
to express future cross-dataset experiments in one YAML without a code branch.

```yaml
folds:
  - name: mpii_to_eyediap
    train:
      - root: /home/hqh/dataset/gaze/MPIIFaceGaze
        image_dir: Image
        label: p00.label
    test:
      - root: /home/hqh/dataset/gaze/EyeDiap
        image_dir: Image
        label: Cluster0.label
```

## Outputs

Each fold writes `save/<exp>/<fold>/result.json` (selected checkpoint + final
test metrics) when it finishes. `summary.json` aggregates every fold present in
the experiment, so cross-validation folds trained in **separate processes**
(e.g. one GPU per fold) still accumulate into one mean. Re-aggregate at any time
without retraining:

```bash
python summarize.py --config config/mpiifacegaze.yaml --exp vits14_mpii
```

TensorBoard data is at `save/<experiment>/<fold>/log/tensorboard`; prediction
grids use green for ground truth and red for prediction. The complete `save/`
tree is ignored by Git. The experiment-level `config.yaml` records CLI
overrides plus the seed and sample limit. Reusing an experiment name with a
different effective configuration is rejected to prevent results and
checkpoints from different runs being mixed.
