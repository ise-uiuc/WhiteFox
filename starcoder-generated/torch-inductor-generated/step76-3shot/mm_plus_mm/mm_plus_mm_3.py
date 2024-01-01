
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        x1 = self.mm(x1, x2)
        x2 = self.mm(x3, x4)
        x3 = x1 + x2
        return x3
# Inputs to the model
x1 = torch.randn(22, 22)
x2 = torch.randn(22, 22)
x3 = torch.randn(22, 22)
x4 = torch.randn(22, 22)
