
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 1, 1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v2 = v2.flatten()
        v3 = torch.tanh(v2)
        v4 = torch.tanh(v3)
        v5 = v4.reshape((v4.shape[0], 1, 7, 2))
        return v5
# Inputs to the model
x = torch.randn(1, 16, 7, 2)
