
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.rand([1, 10, 16, 16])
        self.t2 = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        t3 = self.t1 + x1
        t4 = self.t2(t3)
        t5 = t4 * t4
        return t5
# Inputs to the model
x1 = torch.rand(1, 10, 16, 16)
