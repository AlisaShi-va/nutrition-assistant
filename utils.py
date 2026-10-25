import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from dataset import get_dataloaders