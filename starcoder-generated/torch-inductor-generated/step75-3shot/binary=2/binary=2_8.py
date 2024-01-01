
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 2, 2)
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = x1 - torch.tanh(torch.tanh(v2))
        return v3*torch.sigmoid(v3)*v3 - 1
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
