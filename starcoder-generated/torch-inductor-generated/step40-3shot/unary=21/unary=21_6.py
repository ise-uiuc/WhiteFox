
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 13, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = torch.tanh(self.conv1(x))
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x = torch.randn(16, 1, 55, 55)
