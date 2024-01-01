
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
        self.min_value = kwargs['min_value']
        self.max_value = kwargs['max_value']
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = x2 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(min_value=0, max_value=1)

# Inputs to the model
x1 = torch.randn(2, 10)
