import timm
import detectors


def init_model(device):
  '''
  initialize vit tiny patch16_224 model from timm, no pretraining
  '''
  model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
  model.to(device)
  return model

def get_teacher_model(device):
  '''
  get resnet18 teacher model from timm, with pretrained weights
  '''
  resnet18 = timm.create_model("resnet18_cifar10", pretrained=True)
  #don't want to accidentally update params during training
  for p in resnet18.parameters():
    p.requires_grad = False
  resnet18.to(device)
  return resnet18
