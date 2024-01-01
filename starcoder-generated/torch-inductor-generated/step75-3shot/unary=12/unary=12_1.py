
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.t2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = self.t2(v1)
        v3 =torch.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
