 1: clamp_min
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x, const, min_value):
        v1 = self.linear(x)
        v2 = v1.clamp(min=min_value)
        v3 = v2.clamp_max(max_value=const)
        return v3

# Initializing the model
m = Model()
m

# Inputs to the model
x1 = torch.randn(1, 10)
