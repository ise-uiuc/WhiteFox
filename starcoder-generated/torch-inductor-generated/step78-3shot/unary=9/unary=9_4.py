
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.op2 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.op1(x1)
        v2 = self.op2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
