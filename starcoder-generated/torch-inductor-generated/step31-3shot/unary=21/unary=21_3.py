
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 5, 1, stride=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v3 = self.conv2(v1)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(4, 3, 49, 45)
