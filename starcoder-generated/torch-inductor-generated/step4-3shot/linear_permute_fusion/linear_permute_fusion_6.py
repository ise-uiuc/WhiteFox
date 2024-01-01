
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(2, 4, 3, padding=1)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.permute(0, 2, 3, 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2, 1, 3)
