
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(29, 44, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(44, 29, 1, stride=1)
    def forward(self, x):
        x1 = torch.tanh(x)
        x2 = self.conv1(x1)
        x3 = torch.tanh(x2)
        return self.conv2(x3)
# Inputs to the model
x = torch.randn(2, 29, 1, 1)
