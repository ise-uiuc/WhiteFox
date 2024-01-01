
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = Conv(6, 256, 1, 1, 0, bias=False)
        self.Conv2 = Conv(1, 64, 7, 2, 3, groups=1, bias=False)
    def forward(self, x1):
        t0 = x1
        t1 = self.Conv1.forward(t0)
        t2 = torch.sigmoid(t1)
        t3 = self.Conv2.forward(t2)
        t4 = torch.sigmoid(t3)
        return t4
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
