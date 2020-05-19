import torchvision
import torch
from torchvision import datasets, models, transforms
import os
def load_dataset(Dataset, im_size, cr_size, batch_size):        
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(cr_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = Dataset
        print(Dataset)
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        
        print(image_datasets)
        return dataloaders, class_names, dataset_sizes
