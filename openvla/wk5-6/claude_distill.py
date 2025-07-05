import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preparation
def get_cifar10_loaders(batch_size=128):
  """Get CIFAR-10 data loaders with appropriate transforms"""
  
  # Teacher model transforms (standard ResNet preprocessing)
  teacher_transform = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  # Student model transforms (ViT preprocessing)
  student_transform = transforms.Compose([
    transforms.Resize(224),  # ViT also expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  # For simplicity, using same transform for both (can be modified if needed)
  transform = teacher_transform
  
  train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
  )
  test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
  )
  
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
  
  return train_loader, test_loader

# Teacher Model (Pre-trained ResNet18)
def get_teacher_model():
  """Load pre-trained ResNet18 model from timm"""
  model = timm.create_model('resnet18', pretrained=True, num_classes=10)
  model.to(device)
  return model

# Student Model (Vision Transformer)
def get_student_model():
  """Create ViT model for CIFAR-10"""
  model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
  model.to(device)
  return model

# Distillation Loss
class DistillationLoss(nn.Module):
  """
  Knowledge Distillation Loss combining KL divergence and cross-entropy
  """
  def __init__(self, temperature=4.0, alpha=0.7):
    super(DistillationLoss, self).__init__()
    self.temperature = temperature
    self.alpha = alpha
    self.kl_div = nn.KLDivLoss(reduction='batchmean')
    self.ce_loss = nn.CrossEntropyLoss()
  
  def forward(self, student_logits, teacher_logits, targets):
    # Soften the logits with temperature
    student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
    
    # KL divergence loss (knowledge distillation)
    kd_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
    
    # Cross-entropy loss with ground truth
    ce_loss = self.ce_loss(student_logits, targets)
    
    # Combined loss
    total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
    
    return total_loss, kd_loss, ce_loss

# Training function
def train_student_with_distillation(teacher_model, student_model, train_loader, 
                  test_loader, num_epochs=50, lr=1e-3):
  """Train student model using knowledge distillation"""
  
  # Set teacher to evaluation mode
  teacher_model.eval()
  teacher_model.to(device)
  
  # Set student to training mode
  student_model.train()
  student_model.to(device)
  
  # Loss function and optimizer
  distill_loss = DistillationLoss(temperature=4.0, alpha=0.7)
  optimizer = optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
  
  # Training history
  train_losses = []
  train_accuracies = []
  test_accuracies = []
  
  for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch_idx, (data, targets) in enumerate(pbar):
      data, targets = data.to(device), targets.to(device)
      
      # Forward pass through teacher (no gradients)
      with torch.no_grad():
        teacher_logits = teacher_model(data)
      
      # Forward pass through student
      student_logits = student_model(data)
      
      # Compute distillation loss
      loss, kd_loss, ce_loss = distill_loss(student_logits, teacher_logits, targets)
      
      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # Statistics
      total_loss += loss.item()
      _, predicted = student_logits.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
      
      # Update progress bar
      pbar.set_postfix({
        'Loss': f'{loss.item():.4f}',
        'KD': f'{kd_loss.item():.4f}',
        'CE': f'{ce_loss.item():.4f}',
        'Acc': f'{100.*correct/total:.2f}%'
      })
    
    # Update learning rate
    scheduler.step()
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # Evaluate on test set
    test_acc = evaluate_model(student_model, test_loader)
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, '
        f'Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
  
  return train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader):
  """Evaluate model accuracy on test set"""
  model.eval()
  correct = 0
  total = 0
  
  with torch.no_grad():
    for data, targets in test_loader:
      data, targets = data.to(device), targets.to(device)
      outputs = model(data)
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
  
  accuracy = 100. * correct / total
  return accuracy

def plot_training_history(train_losses, train_accuracies, test_accuracies):
  """Plot training history"""
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
  
  # Plot losses
  ax1.plot(train_losses)
  ax1.set_title('Training Loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.grid(True)
  
  # Plot accuracies
  ax2.plot(train_accuracies, label='Train Accuracy')
  ax2.plot(test_accuracies, label='Test Accuracy')
  ax2.set_title('Model Accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy (%)')
  ax2.legend()
  ax2.grid(True)
  
  plt.tight_layout()
  plt.show()

# Main training script
def main():
  """Main function to run knowledge distillation"""
  print("Starting Knowledge Distillation: ResNet18 (Teacher) -> ViT (Student)")
  
  # Get data loaders
  print("Loading CIFAR-10 dataset...")
  train_loader, test_loader = get_cifar10_loaders(batch_size=128)
  
  # Load teacher model (pre-trained ResNet18)
  print("Loading teacher model (ResNet18)...")
  teacher_model = get_teacher_model()
  
  # Evaluate teacher model performance
  print('devices: ', next(teacher_model.parameters()).device)
  teacher_acc = evaluate_model(teacher_model, test_loader)
  print(f"Teacher model (ResNet18) accuracy: {teacher_acc:.2f}%")
  
  # Create student model (ViT)
  print("Creating student model (ViT)...")
  student_model = get_student_model()
  
  # Evaluate student model before training
  student_acc_before = evaluate_model(student_model, test_loader)
  print(f"Student model (ViT) accuracy before distillation: {student_acc_before:.2f}%")
  
  # Train student with knowledge distillation
  print("Starting knowledge distillation training...")
  train_losses, train_accs, test_accs = train_student_with_distillation(
    teacher_model, student_model, train_loader, test_loader, 
    num_epochs=30, lr=1e-3
  )
  
  # Final evaluation
  final_acc = evaluate_model(student_model, test_loader)
  print(f"\nFinal Results:")
  print(f"Teacher (ResNet18) accuracy: {teacher_acc:.2f}%")
  print(f"Student (ViT) accuracy before distillation: {student_acc_before:.2f}%")
  print(f"Student (ViT) accuracy after distillation: {final_acc:.2f}%")
  print(f"Improvement: {final_acc - student_acc_before:.2f}%")
  
  # Plot training history
  plot_training_history(train_losses, train_accs, test_accs)
  
  return teacher_model, student_model

if __name__ == "__main__":
  # Run the knowledge distillation
  teacher, student = main()
  
  # Save the trained student model
  torch.save(student.state_dict(), 'vit_student_distilled.pth')
  print("Student model saved as 'vit_student_distilled.pth'")