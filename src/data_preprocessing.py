import os

import torch
from torchvision import transforms  # transforms module maintains image size,batch and normalize parameters
from torch.utils.data import DataLoader #  dataloader loads data after DataPreprocessing while training
from torchvision.datasets import ImageFolder
from PIL import Image

class DataPreprocessor:
 def __init__(self,data_dir,batch_size,image_size): #  tensor will be 4D matrix
   """
   :param data_dir:
   :param batch_size:
   :param image_size:
   """
   self.data_dir=data_dir
   self.batch_size=batch_size
   self.image_size=image_size

 def create_data_loaders(self):
  # incoming data will be transformed
  """

  :return:
  """
  data_transforms= {
      'train' : transforms.Compose([
          transforms.Resize((self.image_size, self.image_size)),
          transforms.ToTensor(),
          transforms.Normalize(
              [0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]
          )

      ]),
     'val' : transforms.Compose([
          transforms.Resize((self.image_size, self.image_size)),
          transforms.ToTensor(),
          transforms.Normalize(
              [0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]
          )

      ]),
     'test' : transforms.Compose([  #values from resnet18
          transforms.Resize((self.image_size, self.image_size)),
          transforms.ToTensor(),
          transforms.Normalize(
              [0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225]
          )

      ])
  }
  image_datasets = {
      'train': ImageFolder(os.path.join(self.data_dir,'train'),data_transforms['train']),
      'val':   ImageFolder(os.path.join(self.data_dir,'evaluation'),data_transforms['val']),
      'test':  ImageFolder(os.path.join(self.data_dir,'test'),data_transforms['test'])
  }
  data_loaders ={
      'train': DataLoader(image_datasets['train'],batch_size=self.batch_size,shuffle=True) ,
      'val':   DataLoader(image_datasets['val'],batch_size=self.batch_size,shuffle=False) ,
      'test':  DataLoader(image_datasets['test'],batch_size=self.batch_size,shuffle=False) ,
  }
  return data_loaders