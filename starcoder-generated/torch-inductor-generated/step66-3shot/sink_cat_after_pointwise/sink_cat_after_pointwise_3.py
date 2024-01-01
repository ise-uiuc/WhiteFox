
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t0 = torch.tensor([1.0, 2.0], requires_grad=True)
        self.t1 = torch.tensor([0.5], requires_grad=True)
        self.p0 = torch.nn.Parameter(torch.tensor([0.0]))
        self.p1 = torch.nn.Parameter(torch.tensor([0.0]))
    def forward(self, x):
        x = x * (x + self.p0)
        d0 = self.p1 / x
        ret = torch.cat([self.t0 * d0, self.t1 * d0], dim=0)
        # ret = torch.cat([self.t0 * self.p1 / x, self.t1 * self.p1 / x], dim=0)
        return ret
# Inputs to the model
x = torch.randn(3, 4)
