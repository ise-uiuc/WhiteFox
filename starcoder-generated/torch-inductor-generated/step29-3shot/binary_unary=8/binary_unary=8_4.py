
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.t2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.t3 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = self.t2(x1)
        v3 = self.t3(x1)
        v4 = v1 + v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
