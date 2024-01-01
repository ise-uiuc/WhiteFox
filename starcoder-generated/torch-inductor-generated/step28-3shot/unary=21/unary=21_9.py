
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 2, stride=2)
    def forward(self, x1):
        y1 = self.conv1(x1)
        t1 = torch.tanh(y1)
        z1 = self.conv2(t1)
        s1 = torch.tanh(z1)
        return s1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
