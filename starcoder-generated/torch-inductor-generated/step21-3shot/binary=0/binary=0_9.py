
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ops = torch.nn.ModuleList([
            torch.nn.Conv2d(7, 6, 1, stride=1, padding=0) for _ in range(5)
        ])
    def forward(self, x1, padding1=None, padding2='torch.randn(1, 7, 64, 64)'):
        for op in self.ops:
            v1 = op(x1) + padding2
        return v1
# Inputs to the model
x1 = torch.randn(1, 7, 64, 64)
