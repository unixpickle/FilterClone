"""
Train a small convolutional model on input/output pairs to classify if
the filter actually did anything or not.

Labels are produced by label_changes.ipynb.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

LabelDir = "change_labeled_cartoon"
CkptDir = "ckpt_cartoon"


def main():
    names = [x for x in os.listdir(LabelDir) if not x.startswith(".")]
    random.shuffle(names)
    num_test = 40
    train_names = names[num_test:]
    test_names = names[:num_test]

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    os.makedirs(CkptDir, exist_ok=True)

    def eval_loss(name: str) -> torch.Tensor:
        img1, img2, label = load_pair(name)
        pred = model.apply_to_pair(img1, img2)
        return (
            F.binary_cross_entropy_with_logits(pred[0], torch.tensor([float(label)])),
            (pred.item() > 0) == label,
        )

    for epoch in range(100):
        train_losses, train_corr = [], []
        for name in train_names:
            loss, correct = eval_loss(name)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_losses.append(loss.item())
            train_corr.append(correct)
        test_losses, test_corr = [], []
        for name in test_names:
            with torch.no_grad():
                loss, correct = eval_loss(name)
                test_losses.append(loss.item())
                test_corr.append(correct)
        print(
            f"epoch {epoch}: test_acc={np.mean(test_corr)} train_acc={np.mean(train_corr)} test_loss={np.mean(test_losses)} train_loss={np.mean(train_losses)}"
        )
        with open(f"{CkptDir}/model_{epoch}.pt", "wb") as f:
            torch.save(model.state_dict(), f)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
        )
        self.out_layers = nn.Sequential(
            nn.Linear(32, 1),
        )

    def apply_to_pair(self, img1: Image.Image, img2: Image.Image) -> torch.Tensor:
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")
        img1.thumbnail((256, 256))
        img2.thumbnail((256, 256))
        arr = np.concatenate([np.array(img1), np.array(img2)], axis=-1)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 127.5 - 1
        return self(tensor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spatial_out = self.in_layers(images)
        spatial_out = spatial_out.flatten(2).mean(-1)
        return self.out_layers(spatial_out)


def load_pair(name: str) -> tuple[Image.Image, Image.Image, bool]:
    img1 = Image.open(os.path.join(LabelDir, name, "input.png"))
    img2 = Image.open(os.path.join(LabelDir, name, "output.png"))
    with open(os.path.join(LabelDir, name, "label.json"), "r") as f:
        label = json.load(f)
    return img1, img2, label


if __name__ == "__main__":
    main()
