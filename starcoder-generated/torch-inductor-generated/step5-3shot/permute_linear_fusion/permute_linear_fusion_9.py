
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 30, bias=False)
        self.conv = torch.nn.Conv2d(1, 3, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.linear(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 2)
