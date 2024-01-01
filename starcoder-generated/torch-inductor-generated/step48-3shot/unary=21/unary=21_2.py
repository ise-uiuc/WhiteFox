
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 5, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(5, 1, 3, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x = torch.randn(32, 3, 28, 28)
