
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.mul_ = torch.Tensor([5])
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 3
        v7 = self.relu(v6)
        v8 = v7 * self.mul_
        v9 = self.tanh(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
