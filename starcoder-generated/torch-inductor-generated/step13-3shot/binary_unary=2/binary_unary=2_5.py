
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(100, 512, bias=False)
        self.linear2 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2 - 0.85
        v4 = torch.sigmoid(v3)
        v5 = self.linear2(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 100)
