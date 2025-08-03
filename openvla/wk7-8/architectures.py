import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import RandomSampler, Dataset, DataLoader

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, downsample=False, stride=1):
    super(ResidualBlock, self).__init__()
    self.downsample = downsample
    if self.downsample:
      padding = 0
      self.ds_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    else:
      padding = 1
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels)
    )
    self.relu = nn.ReLU()
    self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, X):
    X_conv = self.conv(X)
    X = self.conv1x1(X)
    if self.downsample:
      X = self.ds_pool(X)
      X = self.ds_pool(X)
    X = X + X_conv
    X = self.relu(X)
    return X

class UnResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, in_shape, upsample=False, stride=1):
    super(UnResidualBlock, self).__init__()
    self.upsample = upsample
    if self.upsample:
      padding = 0
      size_increase1 = (in_shape - 1) * stride + kernel_size
      size_increase2 = (size_increase1 - 1) * stride + kernel_size
      self.up1 = nn.Upsample(size=size_increase1)
      self.up2 = nn.Upsample(size=size_increase2)
    else:
      padding = 1
    self.conv = nn.Sequential(
      nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels)
    )
    self.relu = nn.ReLU()
    self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, X):
    X_conv = self.conv(X)
    X = self.conv1x1(X)
    if self.upsample:
      X = self.up1(X)
      X = self.up2(X)
    X = X + X_conv
    X = self.relu(X)
    return X

class ResNet(nn.Module):
  def __init__(self, dropout):
    super(ResNet, self).__init__()
    self.conv_init = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm2d(num_features=4),
      nn.ReLU()
    )
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    self.resblocks = nn.Sequential(
      ResidualBlock(in_channels=4, out_channels=8, kernel_size=4, downsample=True),
      ResidualBlock(in_channels=8, out_channels=16, kernel_size=4, downsample=True),
      ResidualBlock(in_channels=16, out_channels=32, kernel_size=4, downsample=True),
      ResidualBlock(in_channels=32, out_channels=64, kernel_size=4, downsample=True),
      ResidualBlock(in_channels=64, out_channels=128, kernel_size=2, downsample=True),
      nn.Dropout(dropout),
    )
    self.final_layer = nn.Sequential(
      nn.AvgPool2d(kernel_size=3, padding=0, stride=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(in_features=128, out_features=64, bias=False),
    )
  
  def forward(self, X):
    X = self.conv_init(X)
    X = self.maxpool(X)
    X = self.resblocks(X)
    X = self.final_layer(X)
    return X

class UnResNet(nn.Module):
  def __init__(self, dropout):
    super(UnResNet, self).__init__()
    self.unproject = nn.Sequential(
      nn.Linear(in_features=64, out_features=128),
      nn.Unflatten(dim=-1, unflattened_size=[128, 1, 1]),
      nn.ReLU(),
      nn.Upsample(size=3),
    )
    self.unresblocks = nn.Sequential(
      nn.Dropout(dropout),
      UnResidualBlock(in_channels=128, out_channels=64, kernel_size=2, in_shape=3, upsample=True),
      UnResidualBlock(in_channels=64, out_channels=32, kernel_size=4, in_shape=5, upsample=True),
      UnResidualBlock(in_channels=32, out_channels=16, kernel_size=4, in_shape=11, upsample=True),
      UnResidualBlock(in_channels=16, out_channels=8, kernel_size=4, in_shape=17, upsample=True),
      UnResidualBlock(in_channels=8, out_channels=4, kernel_size=4, in_shape=23, upsample=True), 
    )
    self.rev_maxpool = nn.Upsample(size=30)
    self.rev_conv_init = nn.Sequential(
      nn.ReLU(),
      nn.BatchNorm2d(num_features=4),
      nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=0),       
    )

  def forward(self, X):
    X = self.unproject(X)
    X = self.unresblocks(X)
    X = self.rev_maxpool(X)
    X = self.rev_conv_init(X)
    return X

class ResNetAutoEncoder(nn.Module):
  def __init__(self, dropout=0.2):
    super(ResNetAutoEncoder, self).__init__()
    self.resnet = ResNet(dropout)
    self.unresnet = UnResNet(dropout)
  
  def forward(self, X):
    X_emb = self.resnet(X) #embed X
    X_repr = self.unresnet(X_emb) #reproduce X
    return X_emb, X_repr

#--------------------------------------------------------
class MnistResNet(nn.Module):
  def __init__(self):
    super(MnistResNet, self).__init__()
    self.conv_init = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm2d(num_features=2),
      nn.ReLU()
    )
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    self.resblocks = nn.Sequential(
      ResidualBlock(in_channels=2, out_channels=4, kernel_size=3, downsample=True),
      ResidualBlock(in_channels=4, out_channels=8, kernel_size=3, downsample=True),
      ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, downsample=True),
      ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, downsample=True),
      ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, downsample=True),
      ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, downsample=True),
    )
    self.final_layer = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=128, out_features=32, bias=False),
    )
  
  def forward(self, X):
    X = self.conv_init(X)
    X = self.maxpool(X)
    X = self.resblocks(X)
    X = self.final_layer(X)
    return X
  
class MnistUnResNet(nn.Module):
  def __init__(self):
    super(MnistUnResNet, self).__init__()
    self.unproject = nn.Sequential(
      nn.Linear(in_features=32, out_features=128),
      nn.Unflatten(dim=-1, unflattened_size=[128, 1, 1]),
    )
    self.unresblocks = nn.Sequential(
      UnResidualBlock(in_channels=128, out_channels=64, kernel_size=3, in_shape=1, upsample=True),
      UnResidualBlock(in_channels=64, out_channels=32, kernel_size=3, in_shape=5, upsample=True),
      UnResidualBlock(in_channels=32, out_channels=16, kernel_size=3, in_shape=9, upsample=True),
      UnResidualBlock(in_channels=16, out_channels=8, kernel_size=3, in_shape=13, upsample=True),
      UnResidualBlock(in_channels=8, out_channels=4, kernel_size=3, in_shape=17, upsample=True),
      UnResidualBlock(in_channels=4, out_channels=2, kernel_size=3, in_shape=21, upsample=True),
    )
    self.rev_maxpool = nn.Upsample(size=26)
    self.rev_conv_init = nn.Sequential(
      nn.ReLU(),
      nn.BatchNorm2d(num_features=2),
      nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0),       
    )

  def forward(self, X):
    X = self.unproject(X)
    X = self.unresblocks(X)
    X = self.rev_maxpool(X)
    X = self.rev_conv_init(X)
    return X
  
class MnistResNetAutoEncoder(nn.Module):
  def __init__(self):
    super(MnistResNetAutoEncoder, self).__init__()
    self.resnet = MnistResNet()
    self.unresnet = MnistUnResNet()
  
  def encode(self, X):
    return self.resnet(X)
  
  def decode(self, X_emb):
    return self.unresnet(X_emb)

  def forward(self, X):
    X_emb = self.encode(X) #embed X
    X_repr = self.decode(X_emb) #reproduce X
    return X_emb, X_repr

class MnistResNetVAE(nn.Module):
  def __init__(self):
    super(MnistResNetVAE, self).__init__()
    self.mrnae = MnistResNetAutoEncoder()
    self.resnet = nn.Sequential(*list(self.mrnae.resnet.children())[:-1])
    self.embed_mu = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=128, out_features=32, bias=False),
    )
    self.embed_sig = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=128, out_features=32, bias=False),
    )
    self.unresnet = nn.Sequential(*list(self.mrnae.unresnet.children()))

  def encode(self, X):
    X = self.resnet(X)
    emb_mu = self.embed_mu(X)
    emb_sig = self.embed_sig(X)
    return emb_mu, emb_sig
    
  def decode(self, sample):
    return self.unresnet(sample)
  
  def forward(self, X):
    emb_mu, emb_sig = self.encode(X)
    eps = torch.randn(size=(32,))
    sampled = emb_mu + emb_sig * eps
    X_repr = self.decode(sampled)
    return emb_mu, emb_sig, sampled, X_repr