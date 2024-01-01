
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
        self.bias = torch.nn.Parameter(torch.tensor(3.0))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0.0)
        v4 = torch.clamp_max(v3, 6.0)
        v5 = v4 / 6.0
        return v5
