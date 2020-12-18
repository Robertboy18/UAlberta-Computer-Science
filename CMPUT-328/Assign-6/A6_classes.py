class BasicBlock(nn.Module):
    """
    Heavily inspired from : https://github.com/cdamore/Object-Detection-on-2-digit-MNIST
    Converted from tf to pytorch
    Refernece : https://arxiv.org/abs/1512.03385
    Refernece : https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
    Reference :https://pytorch.org/hub/pytorch_vision_densenet/
    """
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("Conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1))
        self.layer.add_module("Bn", nn.BatchNorm2d(out_channels))

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential()
            self.skip.add_module("Conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1))
            self.skip.add_module("Bn", nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.layer(x)
        out += self.skip(x)
        return F.relu(out)


class DenseNet(nn.Module):
    def __init__(self, block):
        super(DenseNet, self).__init__()

        self.l1 = nn.Sequential()
        self.l1.add_module("Conv", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1))
        self.l1.add_module("Bn", nn.BatchNorm2d(64))
        self.l1.add_module("Relu", nn.ReLU(True))

        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.l2 = nn.Sequential(block(64, 64, 1),block(64, 64, 1))
        self.l3 = nn.Sequential(block(64, 128, 2),block(128, 128, 1))
        self.l4 = nn.Sequential(block(128, 256, 2),block(256, 256, 1))
        self.l5 = nn.Sequential(block(256, 512, 2),block(512, 512, 1))
        self.l7 = nn.Sequential(block(512, 1024, 2),block(1024, 1024, 1))

        self.linear = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(1024, 512),nn.ReLU(True),nn.Linear(512, 37*4+20))

    def forward(self, x):
        x = self.l3(self.l2(self.l1(x)))
        x = self.l5(self.l4(x))
        x = self.pool(self.l7(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x[:,0:20].view(-1, 10, 2), x[:,20:].view(-1, 4, 37)