
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.relu5 = torch.nn.ReLU()
        self.relu6 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.relu1(v3)
        v5 = self.relu2(v4)
        v6 = self.relu3(v5)
        v7 = self.relu4(v3)
        v8 = self.relu5(v7)
        v9 = self.relu6(v8)
        v10 = self.conv3(x1)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
