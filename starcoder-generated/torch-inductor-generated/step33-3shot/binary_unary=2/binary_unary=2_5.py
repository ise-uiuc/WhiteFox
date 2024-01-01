
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = torch.nn.Conv2d(20, 50, 3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = torch.nn.Linear(800, 500)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(500, 10)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.pool1(v1)
        v3 = self.conv2(v2)
        v4 = self.pool2(v3)
        v5 = v4.view(v4.size(0), -1)
        v6 = self.fc1(v5)
        v7 = self.relu(v6)
        v8 = self.fc2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
