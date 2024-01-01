
class Model(torch.nn.Module):
    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self.linear = torch.nn.Linear(20, 7)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

m = Model(-1, 1)
x1 = torch.randn(1, 20)
