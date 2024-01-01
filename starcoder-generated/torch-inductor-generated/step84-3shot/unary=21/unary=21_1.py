
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 6, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = v2 + self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
