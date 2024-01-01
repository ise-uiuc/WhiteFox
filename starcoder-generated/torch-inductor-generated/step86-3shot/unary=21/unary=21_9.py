
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x):
        c1 = torch.tanh(self.conv1(x))
        c2 = torch.tanh(self.conv2(c1))
        return c2
# Inputs to the model
tensor = torch.randn(1, 3, 38, 38)
