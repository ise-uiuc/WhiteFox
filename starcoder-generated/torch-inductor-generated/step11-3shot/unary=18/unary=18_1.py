
class BasicBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if (self.downsample is not None) and (self.last_bn is not None):
            residual = self.downsample(x)
            out += residual
            last_bn = self.last_bn(out)
            out = torch.relu(last_bn)
        elif self.downsample is not None:
            residual = self.downsample(x)
            out += residual
        elif self.last_bn is not None:
            last_bn = self.last_bn(out)
            out = torch.relu(last_bn)
        return out

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.avg_pool = torch.nn.AvgPool2d((4, 4), (2, 3))
        self.dropout1 = torch.nn.Dropout(0.1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(output_size=1)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.avg_pool2 = torch.nn.AvgPool2d((2, 3), (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))

        self.linear1 = torch.nn.Linear(12544, 128)
        self.linear2 = torch.nn.Linear(128, 9)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        v0 = self.conv1(x)
        v1 = torch.relu(v0)
        v2 = self.avg_pool(v1)
        v3 = self.dropout1(v2)
        v4 = self.max_pool(v3)
        v5 = self.dropout2(v4)
        v6 = self.avg_pool2(v5)
        v7 = torch.relu(v6)
        v8 = self.conv2(v7)
        v9 = v8.reshape((v8.size()[0], -1))
        v10 = torch.relu(v9)
        v11 = self.linear1(v10)
        v12 = torch.relu(v11)
        v13 = self.linear2(v12)
        v14 = self.softmax(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
