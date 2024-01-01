
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op_1 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
        self.op_2 = torch.nn.Sigmoid()
        self.op_3 = torch.nn.ReLU()
    def forward(self, x1):
        out = self.op_1(x1)
        op_2 = self.op_2(out)
        out = op_2.mul(out)
        out = self.op_3(out)
        out = out.mul(op_2)
        return out
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
