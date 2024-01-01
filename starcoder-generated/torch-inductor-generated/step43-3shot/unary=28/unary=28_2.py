
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024, bias=True)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, **kwargs)
        v3 = torch.clamp_max(v2, **kwargs)
        return v3

# Initializing the model
m = Model(min_value=-0.5, max_value=0.5)

# Inputs to the model
x1 = torch.randn(1, 1024)
