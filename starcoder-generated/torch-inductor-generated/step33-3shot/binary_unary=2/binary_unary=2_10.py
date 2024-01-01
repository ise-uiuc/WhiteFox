
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)
        self.weight = torch.nn.Parameter(torch.ones(1, 8, 8, 8), requires_grad=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - self.weight - 22
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
