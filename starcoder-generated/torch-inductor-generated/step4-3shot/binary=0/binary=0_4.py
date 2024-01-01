
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1, other=None, x2=None):
        if other == None and x2 == None:
            v1 = torch.tanh(self.conv(x1))
            other = torch.randn(v1.shape)
            v2 = v1 + other
            return v2
        else:
            v1 = self.conv(x1)
            v2 = v1 + other
            v3 = torch.relu(v2)
            return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
