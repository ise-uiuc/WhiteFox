
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 32, (1, 1), bias=False)
        self.norm1 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 32, (1, 7), stride=(1, 2), bias=False)
        self.norm2 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(32, 32, (7, 1), stride=(2, 1), bias=False)
        self.norm3 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.010000000000000009, affine=True)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), bias=False)
        self.norm4 = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True)
        self.relu4 = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.norm1(v1)
        v3 = self.relu1(v2)
        v4 = self.conv2(v3)
        v5 = self.norm2(v4)
        v6 = self.relu2(v5)
        v7 = self.conv3(v6)
        v8 = self.norm3(v7)
        v9 = self.relu3(v8)
        v10 = self.conv4(v9)
        v11 = self.norm4(v10)
        v12 = self.relu4(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 64, 56, 56)
