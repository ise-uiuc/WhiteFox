
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2, bias=False)
        self.relu1 = torch.nn.ReLU6()
        self.conv2 = torch.nn.Conv2d(32, 3, 1, stride=1, padding=0, bias=False)
        self.relu2 = torch.nn.ReLU6()
        self.conv3 = torch.nn.Conv2d(32, 1, 1, stride=1, padding=0)
        self.relu3 = torch.nn.ReLU6()
        self.conv4 = torch.nn.Conv2d(1, 32, 5, stride=1, padding=2, bias=False)
        self.relu4 = torch.nn.ReLU6()
        self.conv5 = torch.nn.Conv2d(32, 3, 1, stride=1, padding=0, bias=False)
        self.relu5 = torch.nn.ReLU6()
        self.conv6 = torch.nn.Conv2d(32, 3, 1, stride=1, padding=0)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = self.relu3(v5)
        v7 = self.conv4(v6)
        v8 = self.relu4(v7)
        v9 = self.conv5(v8)
        v10 = self.relu5(v9)
        v11 = self.conv6(v10)
        v12 = self.relu6(v11)
        return v12
# Inputs to the model
x1 = torch.randn(4, 3, 320, 320)
