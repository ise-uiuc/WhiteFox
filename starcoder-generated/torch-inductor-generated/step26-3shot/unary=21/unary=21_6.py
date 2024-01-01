
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 3)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        return self.conv2(v2)
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
