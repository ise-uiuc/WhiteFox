
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16, bias=False)
 
    def forward(self, x1, min_v, max_v):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_v)
        v3 = torch.clamp_max(v2, max_v)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
