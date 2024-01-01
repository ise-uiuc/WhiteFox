
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v3 = self.conv(x1)
        v1 = self.linear(v3)
        v2 = v1.permute(0, 2, 1)
        v4 = self.tanh(v2)
        return v4
# Inputs to the model
x1 = torch.randn(2, 2, 2, 2)
