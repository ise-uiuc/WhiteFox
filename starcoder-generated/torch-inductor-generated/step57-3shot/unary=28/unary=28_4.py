
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(5, 10, bias=False) 
    def forward(self, x1, **kwargs): 
        self.min_value = min_value
        self.max_value = max_value
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Inputs to the model
x1 = torch.randn(2, 5)
