
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 60, bias=False)

    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 + 3
        x4 = torch.clamp_min(x3, 0)
        x5 = torch.clamp_max(x4, 6)
        x6 = x5 / 6
        return x6

m = Model()

x1 = torch.rand(10, 20)
