
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, padding=2)
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, padding=2)
        self.fc1 = torch.nn.Linear(50 * 2 * 2, 1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.pool(v2)
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        v6 = v5.view(-1, 50 * 2 * 2)
        v7 = self.fc1(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x = torch.randn(1, 3, 30, 30)
