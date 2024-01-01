
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, x1, min_value, max_value):
        v1 = torch.matmul(x1, self.weight)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model with dummy weights
m = Model()
m.weight = torch.nn.Parameter(torch.zeros(3, 5))

# Inputs to the model
x1 = torch.randn(1, 3)
__min_value__ = 5
__max_value__ = 7
