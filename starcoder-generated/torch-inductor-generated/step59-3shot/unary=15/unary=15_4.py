
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu1 = torch.nn.PReLU(3, 3)
        self.prelu2 = torch.nn.PReLU(3, 1)
    def forward(self, x1):
        v1 = self.prelu1(x1)
        v2 = self.prelu2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 72, 72)
