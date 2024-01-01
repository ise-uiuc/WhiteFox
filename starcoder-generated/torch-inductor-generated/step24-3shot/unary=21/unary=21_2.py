
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = torch.tanh(v1 + v2)
        return v3
# Inputs to the model
x = torch.randn(1, 128, 28, 28)
