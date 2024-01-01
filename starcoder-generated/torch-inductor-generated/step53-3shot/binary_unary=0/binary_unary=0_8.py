
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.squeeze
        self.op2 = torch.mean
        self.op3 = torch.add
    def forward(self, x):
        v1 = self.op1(x)
        v2 = self.op2(v1)
        v3 = self.op3(v2)
        return v3
# Inputs to the model
x = torch.randn(16, 3, 64, 64)
