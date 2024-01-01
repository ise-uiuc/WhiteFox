
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(20, 64, bias=False)
        self.conv = torch.nn.Conv2d(44, 2, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.linear_1(x)
        v2 = v1 - 55
        if v2.ndim == 1:
            v3 = v2.sum()
        else:
            v3 = v2.sum((-1,))
        v4 = torch.relu(v3)
        v5 = self.conv(v4)
        v6 = v5 - 700
        return v6
# Inputs to the model
x = torch.randn(1, 20)
