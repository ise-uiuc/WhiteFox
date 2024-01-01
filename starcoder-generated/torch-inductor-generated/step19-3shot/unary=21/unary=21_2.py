
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv2d(3, 13, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(12, 10, 3, stride=2, padding=1, groups=4)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.tanh(v1)
        return self.conv2(v2)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
