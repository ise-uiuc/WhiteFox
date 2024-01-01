
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, dilation=1, padding=0, stride=1, groups=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = torch.tanh(v2).detach()
        return v3
# Inputs to the model
x = torch.randn(1, 3, 60, 60)
