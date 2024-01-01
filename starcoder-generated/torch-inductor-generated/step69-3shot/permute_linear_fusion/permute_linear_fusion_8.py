
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(2, 3, 3)
        self.unfold = torch.nn.Unfold((2, 2))
    def forward(self, x1):
        v1 = self.unfold(x1)
        v2 = self.linear(v1)
        v3 = v2.permute(0, 2, 1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
