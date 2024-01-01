
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64, bias=False)
 
    def forward(self, x, *, min_value, max_value):
       v1 = self.linear(x)
       v2 = torch.clamp_min(v1, min_value)
       return torch.clamp_max(v2, max_value)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
