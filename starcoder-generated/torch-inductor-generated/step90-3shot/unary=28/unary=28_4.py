
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 64)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, kwargs.get('min_value', None))
        v3 = torch.clamp_max(v2, kwargs.get('max_value', None))
        return v3

# Initializing the model
m = Model()
m.eval()

# Inputs to the model
x1 = torch.randn(1, 4)
min_v = 0.0
max_v = 1.0
