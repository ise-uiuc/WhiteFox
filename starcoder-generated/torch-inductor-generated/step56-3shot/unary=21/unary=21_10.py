
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 1, 1)
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
input = torch.randn(1, 64, 1, 1)
# Model Ends

# Model begins
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x0 = torch.randn(1, 2)
# Model Ends

# Model begins
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(2)
        self.conv2 = torch.nn.Conv2d(2, 3, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = self.bn1(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = torch.tanh(v5)
        return v3
# Inputs to the model
input = torch.randn(1, 3, 28, 28)
