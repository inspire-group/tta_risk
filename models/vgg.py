'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

class Normalized_VGG_CIFAR100(VGG):
    def __init__(self, device="cuda"):
        super(Normalized_VGG_CIFAR100, self).__init__('VGG19',100)
        self.mu = torch.Tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).float().view(3, 1, 1).to(device)
        self.sigma = torch.Tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).float().view(3, 1, 1).to(device)

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Normalized_VGG_CIFAR100, self).forward(x)


class Normalized_VGG_CIFAR10(VGG):
    def __init__(self, device="cuda"):
        super(Normalized_VGG_CIFAR10, self).__init__('VGG19',10)
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).to(device)
        self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(3, 1, 1).to(device)

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Normalized_VGG_CIFAR10, self).forward(x)