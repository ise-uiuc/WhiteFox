
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t0 = torch.tensor([[1, 2, 3]])
        self.linear = torch.nn.Linear(2, 2)
        self.convbn = torch.nn.BatchNorm2d(2, affine=True)
    def forward(self, x1):
        t1 = x1+self.t0
        t2 = t1.permute(2, 1, 0)
        v1 = self.convbn(t2)
        v2 = v1.permute(2, 1, 0)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
