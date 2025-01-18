"""
Apply a classifier trained by train_change_clf.py to a dataset.
"""

import argparse
import os
import shutil

import torch
from PIL import Image
from tqdm.auto import tqdm
from train_change_clf import Model

LabelDir = "change_labeled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="ckpt/model_99.pt")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filtered_dir", type=str, required=True)
    args = parser.parse_args()

    model = Model()
    with open(args.model_ckpt, "rb") as f:
        model.load_state_dict(torch.load(f))

    os.makedirs(args.filtered_dir, exist_ok=True)
    out_names = os.listdir(args.output_dir)
    kept = 0
    total = 0
    for out_name in tqdm(out_names):
        if out_name.startswith("."):
            continue
        out_path = os.path.join(args.output_dir, out_name)
        try:
            img1 = Image.open(os.path.join(args.input_dir, out_name))
            img2 = Image.open(out_path)
        except KeyboardInterrupt:
            raise
        except:
            continue
        if img1.size != img2.size:
            continue
        try:
            logit = model.apply_to_pair(img1, img2).item()
        except KeyboardInterrupt:
            raise
        except:
            # Might happen due to very weird aspect ratio
            continue
        if logit > 0:
            shutil.copy(out_path, os.path.join(args.filtered_dir, out_name))
            kept += 1
        total += 1
    print(f"kept a total of {kept}/{total} ({100*kept/total:.02f}%)")


if __name__ == "__main__":
    main()
