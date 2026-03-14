"""Softmax - PyTorch Baseline"""
import torch
import torch.nn.functional as F


def softmax_pytorch(X: torch.Tensor) -> torch.Tensor:
    return F.softmax(X, dim=-1)
