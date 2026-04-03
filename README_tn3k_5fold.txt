TN3K 5-fold cross-validation usage
==================================

Files in this package:
1. train_5fold.py
2. trainer_5fold.py
3. dataset_tn3k_5fold.py

What this version does:
- Combines the official TN3K train.txt and test.txt into one full sample pool.
- Keeps the original physical directory structure unchanged:
    root_path/
      train/images, train/masks
      test/images,  test/masks
- Builds deterministic 5-fold splits from that combined pool.
- Trains one model per fold.
- Evaluates on the validation fold every epoch.
- Saves:
    fold_k/best_model.pth
    fold_k/last_model.pth
    fold_k/log.txt
    fold_k/train_records.txt
    fold_k/val_records.txt
    fold_k/fold_result.json
- After all folds, writes cv_results.json with mean/std Dice.

How to place the files:
- Put train_5fold.py, trainer_5fold.py, dataset_tn3k_5fold.py in the same directory as your original train.py/config.py.
- Keep your existing networks/ folder unchanged.
- Keep utils.py unchanged.

Single-GPU example:
python train_5fold.py \
  --dataset TN3K \
  --root_path /path/to/TN3K \
  --list_dir /path/to/lists/TN3K \
  --cfg /path/to/config.yaml \
  --output_dir /path/to/output_tn3k_5fold \
  --img_size 384 \
  --max_epochs 100 \
  --batch_size 6 \
  --n_class 2 \
  --base_lr 1e-4

Run only one fold, e.g. fold 0:
python train_5fold.py \
  --dataset TN3K \
  --root_path /path/to/TN3K \
  --list_dir /path/to/lists/TN3K \
  --cfg /path/to/config.yaml \
  --output_dir /path/to/output_tn3k_5fold \
  --img_size 384 \
  --max_epochs 100 \
  --batch_size 6 \
  --n_class 2 \
  --base_lr 1e-4 \
  --fold 0

Multi-GPU DDP example (3 GPUs):
torchrun --nproc_per_node=3 train_5fold.py \
  --dataset TN3K \
  --root_path /path/to/TN3K \
  --list_dir /path/to/lists/TN3K \
  --cfg /path/to/config.yaml \
  --output_dir /path/to/output_tn3k_5fold \
  --img_size 384 \
  --max_epochs 100 \
  --batch_size 4 \
  --n_class 2 \
  --base_lr 1e-4

Notes:
- In cross-validation, the official test set is no longer used as a final standalone test set.
- The official train + test lists are merged and then re-split into folds.
- The paper result should report mean ± std across folds.
