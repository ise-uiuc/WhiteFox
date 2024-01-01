
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x):
        n1 = self.conv1(x)
        n2 = torch.tanh(n1)
        n3 = self.conv2(n1)
        n4 = torch.tanh(n3)
        return n4
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
