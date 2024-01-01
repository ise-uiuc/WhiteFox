
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.b = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.g = torch.nn.Identity()
    def forward(self, x1, other=1):
        v1 = self.g(self.a(x1))
        if other == 1:
            other = self.b(x1)
        v2 = other + v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
