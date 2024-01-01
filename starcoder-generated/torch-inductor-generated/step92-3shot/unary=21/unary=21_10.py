
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=(2, 2))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1
        v3 = torch.tanh(v2)
        v4 = v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
