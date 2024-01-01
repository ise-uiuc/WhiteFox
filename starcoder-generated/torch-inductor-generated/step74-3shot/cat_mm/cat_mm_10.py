
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(100, 100)
        self.input = torch.randn(20, 100)
    def forward(self):
        v1 = torch.mm(self.input, self.layer.weight)
        v2 = torch.mm(self.input, self.layer.weight)
        v3 = torch.mm(self.input, self.layer.weight)
        return torch.cat([v1, v2, v3], 1)
