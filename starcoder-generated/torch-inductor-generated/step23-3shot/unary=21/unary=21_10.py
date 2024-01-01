
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 1)
        self.linear = torch.nn.Linear(940, 84, bias=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.view(-1, 940)
        z1 = self.linear(v2)
        z2 = torch.tanh(z1)
        return z2
# Inputs to the model
x1 = torch.randn(255, 3, 256, 256)
