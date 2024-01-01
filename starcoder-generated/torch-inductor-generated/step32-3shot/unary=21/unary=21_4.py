
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 3, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 1, 33, 33)
