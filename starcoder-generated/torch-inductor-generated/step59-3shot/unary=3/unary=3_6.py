
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 * 1.1
        v3 = v1 * 2.1
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 35, 35)
