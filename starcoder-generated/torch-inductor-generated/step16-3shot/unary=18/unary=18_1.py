
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(3, affine=False)
        self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
        self.bn3 = torch.nn.BatchNorm2d(3, affine=False)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.relu3 = torch.nn.ReLU(inplace=False)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=1)
        self.add = torch.add
        self.dropout1 = torch.nn.Dropout(0.2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.relu1(v2)
        v4 = self.pool1(v3)
        v5 = self.conv2(v4)
        v6 = self.bn2(v5)
        v7 = self.relu2(v6)
        v8 = self.pool2(v7)
        v9 = v7.flatten(start_dim=1, end_dim=-1)
        v10 = self.dropout1(v9.float())
        v11 = self.conv3(v8)
        v12 = self.bn3(v11)
        v13 = self.relu3(v12)
        v14 = v13 + v2
        return v14.int()
# Inputs to the model
x1 = torch.randn(3, 3, 224, 224)
