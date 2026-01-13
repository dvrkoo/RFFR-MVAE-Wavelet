import json
import math
import pandas as pd
import numpy as np
import torch
import os
import sys
from glob import glob
from configs.config import config
import shutil
from tqdm import tqdm
import random


def output_to_patch(output, row, col):
    # output: [b, c, h, w]
    # row / col:  [b, 1]
    W, H = output.shape[-1], output.shape[-2]
    offsets = (torch.cat([col, row], dim=1) * 16 + 16 * 1.5).cuda()
    h, w = 48, 48
    xs = (torch.arange(0, w) - (w - 1) / 2.0).cuda()
    ys = (torch.arange(0, h) - (h - 1) / 2.0).cuda()

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

    offsets_grid = offsets[:, None, None, :] + grid[None, ...]

    # normalised grid  to [-1, 1]
    offsets_grid = (
        offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])
    ) / offsets_grid.new_tensor([W / 2, H / 2])

    return torch.nn.functional.grid_sample(
        output, offsets_grid, mode="bilinear", padding_mode="zeros", align_corners=None
    )


def calc_auc(y_labels, y_scores):
    # y_scores = y_scores / max(y_scores)
    f = list(zip(y_scores, y_labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    auc = (np.sum(rankList) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)
    return auc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mkdirs():
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    if not os.path.exists(config.best_model_path):
        os.makedirs(config.best_model_path)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)


def test_checkpoint_save():
    print("\n[Pre-flight] Testing checkpoint save capability...")

    try:
        test_state = {
            "epoch": 0,
            "test": "checkpoint_write_test",
            "data": torch.randn(100, 100),
        }

        test_filepath = config.checkpoint_path + "_test_checkpoint.pth.tar"
        temp_filepath = test_filepath + ".tmp"

        stat = shutil.disk_usage(config.checkpoint_path)
        free_gb = stat.free / (1024**3)

        if stat.free < 5 * 1024**3:
            print(
                f"[ERROR] Insufficient disk space: {free_gb:.2f} GB free (need at least 5 GB)"
            )
            raise RuntimeError(
                f"Insufficient disk space for checkpointing: {free_gb:.2f} GB available"
            )

        torch.save(test_state, temp_filepath)

        if not os.path.exists(temp_filepath):
            raise RuntimeError(f"Failed to create test checkpoint at {temp_filepath}")

        file_size_kb = os.path.getsize(temp_filepath) / 1024

        os.rename(temp_filepath, test_filepath)

        if not os.path.exists(test_filepath):
            raise RuntimeError("Atomic rename failed during test")

        loaded_state = torch.load(test_filepath)
        if loaded_state.get("test") != "checkpoint_write_test":
            raise RuntimeError("Test checkpoint corrupted: failed to load correctly")

        os.remove(test_filepath)

        print(f"[Pre-flight] Checkpoint save test PASSED")
        print(f"[Pre-flight] - Write test: {file_size_kb:.1f} KB written successfully")
        print(f"[Pre-flight] - Atomic rename: working")
        print(f"[Pre-flight] - Read/verify: working")
        print(f"[Pre-flight] - Available space: {free_gb:.2f} GB")
        return True

    except Exception as e:
        print(f"[ERROR] Checkpoint save test FAILED: {e}")
        print(
            f"[ERROR] Training cannot proceed - fix checkpoint directory issues first"
        )

        for path in [temp_filepath, test_filepath]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

        raise RuntimeError(f"Checkpoint save test failed: {e}")


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return "%2d hr %02d min" % (hr, min)
    elif mode == "sec":
        t = int(t)
        min = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (min, sec)
    else:
        raise NotImplementedError


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = "w"
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def save_checkpoint(
    save_list, model, optimizer, model_config=None, filename="_checkpoint.pth.tar"
):
    import time

    epoch = save_list[0]
    current_loss = save_list[1]
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "adam_dict": optimizer.state_dict(),
    }

    if model_config is not None:
        state["config"] = model_config

    filepath = config.checkpoint_path + filename
    best_filepath = (
        config.best_model_path
        + "best_loss_"
        + str(round(current_loss, 5))
        + "_"
        + str(epoch)
        + ".pth.tar"
    )

    estimated_size = sum(p.numel() * p.element_size() for p in model.parameters())
    estimated_size += sum(
        v.numel() * v.element_size()
        for v in optimizer.state_dict()["state"].values()
        if isinstance(v, torch.Tensor)
    )
    estimated_size_mb = estimated_size / (1024 * 1024)

    print(
        f"\n[Checkpoint] Saving epoch {epoch}, estimated size: {estimated_size_mb:.1f} MB"
    )

    def check_disk_space(path, required_bytes):
        try:
            stat = shutil.disk_usage(os.path.dirname(path))
            free_gb = stat.free / (1024**3)
            required_gb = required_bytes / (1024**3)
            print(
                f"[Checkpoint] Free space: {free_gb:.2f} GB, required: {required_gb:.2f} GB"
            )
            if stat.free < required_bytes * 3:
                print(
                    f"[WARNING] Low disk space! Free: {free_gb:.2f} GB, need ~{required_gb * 3:.2f} GB for safety"
                )
            return stat.free >= required_bytes
        except Exception as e:
            print(f"[WARNING] Could not check disk space: {e}")
            return True

    def atomic_save(state, target_path, max_retries=3):
        for attempt in range(max_retries):
            try:
                if not check_disk_space(target_path, estimated_size * 2):
                    raise RuntimeError(
                        f"Insufficient disk space for checkpoint at {target_path}"
                    )

                temp_path = target_path + ".tmp"
                torch.save(state, temp_path)

                if os.path.exists(temp_path):
                    actual_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                    print(f"[Checkpoint] Temp file written: {actual_size_mb:.1f} MB")
                    os.rename(temp_path, target_path)
                    print(f"[Checkpoint] Successfully saved to {target_path}")
                    return True
                else:
                    raise RuntimeError(f"Temp file not created at {temp_path}")

            except Exception as e:
                print(
                    f"[ERROR] Checkpoint save attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"[Checkpoint] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"[ERROR] Failed to save checkpoint after {max_retries} attempts"
                    )
                    raise
        return False

    atomic_save(state, filepath)

    try:
        if os.path.exists(filepath):
            shutil.copy(filepath, best_filepath)
            print(f"[Checkpoint] Copied to best model: {best_filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to copy to best model path: {e}")


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def save_code(time, runhash):
    directory = "../history/" + time + "_" + runhash + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filepath in config.save_code:
        shutil.copy(filepath, directory + filepath.split("/")[-1])
