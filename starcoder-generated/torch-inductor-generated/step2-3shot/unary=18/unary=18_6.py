
# Description: conv2d + batchnorm2d + ReLU6 + conv2d + batchnorm2d + ReLU (pattern: 3 + 3 + 3) + MaxPool2d with kernel size 2 and stride 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(11, 11, 5, stride=1, padding=(2,2), bias=False)
        self.b1 = torch.nn.BatchNorm2d(11)
        self.g1 = torch.nn.ReLU6()
        