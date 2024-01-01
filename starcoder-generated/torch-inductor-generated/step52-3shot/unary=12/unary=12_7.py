
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=2, padding=2, dilation=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.tanh(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.Tensor(1, 3, 64, 64)
