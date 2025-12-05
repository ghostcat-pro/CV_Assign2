# Repository Guidelines

## Project Structure & Module Organization
- `models/` holds all segmentation architectures (UNet-ResAttn variants, DeepLabV3, SUIM-Net) plus shared blocks.
- `datasets/` contains loaders and augmentations tailored for SUIM RGB masks.
- `training/` provides the training loop, loss functions, metrics, and utilities shared across scripts.
- Top-level scripts: `organize_suim_dataset.py` and `create_splits.py` prepare data; `main_train.py`, `train_unet_v2.py`, `train_unet_v3.py`, and `train_all_models.sh` handle training; `evaluate.py`, `evaluate_with_fscore.py`, and `test_models.py` cover evaluation and smoke tests.
- Generated artifacts (`data/`, `checkpoints/`, `logs/`, `raw_suim/`) are git-ignored; recreate them locally via the setup scripts.

## Build, Test, and Development Commands
- Create environment: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Prepare data after placing SUIM into `raw_suim/`: `python organize_suim_dataset.py` then `python create_splits.py`.
- Quick model sanity check (CPU-safe): `python test_models.py`.
- Train all models sequentially: `bash train_all_models.sh`; train only V3: `python train_unet_v3.py --epochs 50 --batch_size 6`.
- Evaluate all models with IoU and F-score: `python evaluate_with_fscore.py`; evaluate one model: `python evaluate.py --model <name> --checkpoint checkpoints/<file>.pth`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; prefer explicit names for layers, losses, and datasets.
- Match existing module structure: model variants in `models/`, shared helpers in `training/`, transforms in `datasets/`.
- Use type hints where practical and keep lines ≤100 chars in new code to match current style.
- Function/class names: `CamelCase` for modules, `snake_case` for functions/variables; CLI flags stay kebab-case when using argparse.

## Testing Guidelines
- Primary smoke test: `python test_models.py` (verifies model construction and forward passes).
- For training changes, run at least a short V3 epoch locally to ensure loss/metrics move; document any reduced batch sizes for GPU limits.
- Add targeted unit tests for new utilities under `training/` or `datasets/` when logic is non-trivial; keep fixtures small to avoid committing data.
- Attach metric outputs (mIoU/F-score) or log snippets in PRs when altering training, loss, or evaluation logic.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject lines (e.g., "Add focal loss tweak to V3"); group related changes per commit.
- PRs: include a brief summary, key commands run, and before/after metrics or screenshots for training/eval changes; link issues when applicable.
- Do not commit datasets, checkpoints, or large logs. Document expected paths (e.g., `data/images/train`) and how to regenerate artifacts.

## Data & Security Notes
- Keep SUIM data under `raw_suim/` and generated splits under `data/`; avoid embedding paths to personal directories in code.
- When sharing results, reference checkpoint filenames only (not full local paths) to prevent leaking local environment details.
