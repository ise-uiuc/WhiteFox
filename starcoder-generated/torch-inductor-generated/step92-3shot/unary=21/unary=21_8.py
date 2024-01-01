
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
    def forward(self, x):
        v1 = self.conv1(x).to(torch.float)
        v2 = torch.tanh(v1).to(torch.float)
        return v2
# Inputs to the model
x = torch.randn(1, 64, 64, 64)
