import numpy as np
# from pycocotools.coco import COCO
import torch
import torchvision
import torchvision.transforms as transforms


data_dir = '../../../data'
im_size = 32
download = True

# coco = COCO(annotation_file='{}/coco/annotations/instances_train2017.json'.format(data_dir))


def gen_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False, num_workers=2
    )
    print('Number of data samples:', len(dataset))
    train = iter(dataloader).next()[0]
    print('Data shape:', train.shape)
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    print('Mean: ', mean)
    print('Std: ', std)


if __name__ == '__main__':
    # dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transforms.Compose([transforms.ToTensor()]))
    # dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transforms.Compose([transforms.ToTensor()]))
    # dataset = torchvision.datasets.VOCSegmentation( root=data_dir, image_set='train', download=download, transform=transforms.Compose([transforms.Resize((im_size, im_size)), transforms.ToTensor()]), target_transform=transforms.Compose([transforms.Resize((im_size, im_size)), transforms.ToTensor()]))
    dataset = torchvision.datasets.SVHN(
        root=data_dir, split='test', download=download, transform=transforms.Compose([transforms.ToTensor()]))
    # dataset = torchvision.datasets.CocoDetection(root = '{}/coco/train2017/'.format(data_dir), annFile='{}/coco/annotations/instances_train2017.json'.format(data_dir), transform=transforms.Compose([transforms.Resize((im_size, im_size)), transforms.ToTensor()]))
    # dataset = torchvision.datasets.ImageNet(
    #     root=data_dir,
    #     transform=transforms.Compose(
    #         [transforms.Resize((im_size, im_size)), transforms.ToTensor()]),
    # )

    gen_mean_std(dataset)
