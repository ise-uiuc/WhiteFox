
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(32, 8)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 + 3
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v4 * 0.16666666666666666
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
__output1__ = m(x1)
x2 = torch.randn(2, 32)
__output2__ = m(x2)
x3 = torch.randn(8, 32)
__output3__ = m(x3)
x4 = torch.randn(16, 32)
__output4__ = m(x4)
x5 = torch.randn(64, 32)
__output5__ = m(x5)

