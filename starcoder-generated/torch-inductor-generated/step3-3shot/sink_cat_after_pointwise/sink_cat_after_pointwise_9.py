
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v4 = torch.arange(1).int().view(1, 1) + 1
    def forward(self, x):
        x1 = x + self.v4
        x2 = self.v4 + x
        x3 = torch.abs(x2)
        y = torch.cat((x1, x3), dim=1)
        return y
# Inputs to the model
x = torch.randn(1, 3)
