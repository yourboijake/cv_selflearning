import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from load_cifar import get_cifar10_data, get_soft_data
from vit import init_model, get_teacher_model
import datetime

#accuracy of classification
def eval_model(model, testloader, device):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, labels in testloader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      preds = outputs.argmax(dim=-1)
      correct += preds.eq(labels).sum()
      total += inputs.shape[0]
  return correct / float(total)

def train(model, trainloader, testloader, device, teacher_model=None, temperature=0.5, alpha=0.5):
  optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  ce_loss_criterion = nn.CrossEntropyLoss()
  kld_loss_criterion = nn.KLDivLoss(reduction='batchmean')
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

  train_epoch_losses, train_epoch_acc, test_epoch_acc = [], [], []
  epoch_values, iters, train_losses, train_acc = [], [], [], []

  for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    correct = 0
    total = 0
    model.train()
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
      #forward and backward pass
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      if teacher_model:
        with torch.no_grad():
          soft_labels = F.softmax(teacher_model(inputs) / temperature, dim=1) #get soft labels from teacher model
        soft_preds = F.log_softmax(outputs / temperature, dim=1)
        kld_loss = kld_loss_criterion(soft_preds, soft_labels) * (temperature ** 2)
        ce_loss = ce_loss_criterion(outputs, labels)
        loss = alpha * kld_loss + (1 - alpha) * ce_loss #combine CELoss and KL-Divergence Loss
      else:
        loss = ce_loss_criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      #update running statistics
      total_loss += loss.item()
      preds = outputs.argmax(dim=-1)
      correct += preds.eq(labels).sum()
      total += labels.size(0)

      #update progress bar
      pbar.set_postfix({
        'Loss': f'{loss.item():.4f}',
        'Acc': f'{100.*correct/total:.2f}%'
      })
      epoch_values.append(epoch)
      iters.append(batch_idx)
      train_losses.append(loss.item())
      train_acc.append(correct / float(total))
    
    # update learning rate
    scheduler.step()
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    test_acc = eval_model(model, testloader, device)
    
    train_epoch_losses.append(epoch_loss)
    train_epoch_acc.append(epoch_acc)
    test_epoch_acc.append(test_acc)
    
    print(f'Epoch {epoch+1}: Train CELoss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')

  results = {
    'train_epoch_losses': train_epoch_losses, 
    'train_epoch_acc': train_epoch_acc, 
    'test_epoch_acc': test_epoch_acc,
    'epoch_values': epoch_values, 
    'iters': iters, 
    'train_losses': train_losses, 
    'train_acc': train_acc
  }
  return results

def save_stats(results, output_file_prefix, ts):
  #save batch-level stats
  with open(f'./data/{output_file_prefix}_batchstats_{ts}.csv', 'w') as f:
    for i in range(len(results['train_losses'])):
      row = f'{results['epoch_values'][i]},{results['iters'][i]},{results['train_losses'][i]},{results['train_acc'][i]}\n'
      f.write(row)

  #save epoch-level stats
  with open(f'./data/{output_file_prefix}_epochstats_{ts}.csv', 'w') as f:
    for i in range(len(results['train_epoch_losses'])):
      row = f'{i},{results['train_epoch_losses'][i]},{results['train_epoch_acc'][i]},{results['test_epoch_acc'][i]}\n'
      f.write(row)

if __name__ == '__main__':
  #define hyperparams/training constants
  BATCH_SIZE = 128
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-2
  NUM_EPOCHS = 10

  #retrieve device
  if torch.cuda.is_available():
    device = torch.device('cuda')
  elif torch.xpu.is_available():
    device = torch.device('xpu')
  else:
    device = torch.device('cpu')
  print('using device:', device)
  print('\n')

  #create models
  vit_normal = init_model(device)
  vit_student = init_model(device)
  resnet_teacher = get_teacher_model(device)
  vit_param_count = sum([p.numel() for p in vit_normal.parameters()])
  resnet_param_count = sum([p.numel() for p in resnet_teacher.parameters()])
  print(f'vit_normal and vit_student parameter count: {vit_param_count:,}')
  print(f'resnet18 teacher parameter count: {resnet_param_count:,}')
  print('\n')

  #retrieve Dataloader objects
  trainset, testset, trainloader, testloader = get_cifar10_data(batch_size=128)

  #show initial performance of model before training
  print('evaluating accuracy of default parameters on CIFAR10 test set...')
  acc = eval_model(vit_normal, testloader, device)
  print(f'initial accuracy {acc:,}')


  #train vit_normal, store results
  print('training vit using normal pre-training approach')
  results = train(vit_normal, trainloader, testloader, device)
  ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  save_stats(results, 'vit_normal', ts)
  torch.save(vit_normal.state_dict(), f'./models/vit_normal_{ts}.pt')
  
  #use train vit_student with teacher model
  print('training vit using distillation/student-teacher approach')
  results = train(vit_normal, trainloader, testloader, device, teacher_model=resnet_teacher)
  ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  save_stats(results, 'vit_student', ts)
  torch.save(vit_student.state_dict(), f'./models/vit_student_{ts}.pt')

