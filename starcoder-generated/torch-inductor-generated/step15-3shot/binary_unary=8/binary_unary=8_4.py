
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 7, 1)
        self.t = torch.tensor(1000)
        self.register_buffer('t2', torch.tensor(100))
        self.t3 = torch.tensor(1000)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.t * v1
        v3 = self.t2 * v2
        v4 = self.t3 * v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 500, 500)
