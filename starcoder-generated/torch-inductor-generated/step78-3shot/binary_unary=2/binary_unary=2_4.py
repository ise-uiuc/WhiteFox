
class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 8, 3, stride=2, padding=1) 
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.relu(x3)
        return x4

class Model_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.conv_2 = torch.nn.Conv2d(10, 8, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        x2 = self.conv_2(x1)
        x3 = self.conv_1(x2)
        x4 = self.relu(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
