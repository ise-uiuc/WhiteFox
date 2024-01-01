
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 1, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.tanh(v2)
        return v3.detach()
# Inputs to the model
x = torch.randn(64, 3, 64, 64)
