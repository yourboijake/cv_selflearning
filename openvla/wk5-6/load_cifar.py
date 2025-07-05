import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from vit import get_teacher_model
from tqdm import tqdm

def get_cifar10_data(batch_size):
  """Get CIFAR-10 data loaders with appropriate transforms"""
  transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
  ])
  
  trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
  )
  testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
  )
  
  trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
  
  return trainset, testset, trainloader, testloader

class MultiLabelDataset(Dataset):
  def __init__(self, data, labels1, labels2, transform=None):
    if len(data) != len(labels1) or len(data) != len(labels2):
      raise ValueError("All input lists must have the same length.")

    self.data = data
    self.labels1 = labels1
    self.labels2 = labels2
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample = self.data[idx]
    label1 = self.labels1[idx]
    label2 = self.labels2[idx]

    if self.transform:
      sample = self.transform(sample)

    return sample, label1, label2

def get_soft_data(teacher, trainloader, batch_size, device):
  #datasets = []
  imgs, hard_labels, soft_labels = [], [], []
  teacher.eval()
  pbar = tqdm(trainloader, desc=f'running teacher on training set...')
    

  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(pbar):
    #for i, data in enumerate(trainloader, 0):
      #inputs, labels = data
      inputs = inputs.to(device)
      outputs = teacher(inputs)
      imgs.append(inputs.cpu())
      hard_labels.append(labels)
      soft_labels.append(outputs.cpu())

      #update progress bar
      pbar.set_postfix()
      #if batch_idx > 3: break

  soft_trainset = MultiLabelDataset(
    torch.concat(imgs, dim=0),
    torch.concat(hard_labels, dim=0),
    torch.concat(soft_labels, dim=0),
  )
  soft_trainloader = torch.utils.data.DataLoader(
    soft_trainset,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2,
  )
  return soft_trainset, soft_trainloader

if __name__ == '__main__':
  device = torch.device('xpu')
  teacher_model = get_teacher_model(device)
  trainset, testset, trainloader, testloader = get_cifar10_data(128)
  soft_trainset, soft_trainloader = get_soft_data(teacher_model, trainloader, 128, device)
  torch.save(soft_trainset, 'soft_trainset.pt')