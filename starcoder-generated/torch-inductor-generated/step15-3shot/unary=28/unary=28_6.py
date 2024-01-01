
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(8, 64)
 
    def forward(x2, **kwargs):
        v1 = self.linear(x2, **kwargs)
        v2 = torch.clamp(v1, min_value=kwargs['min_value'], max_value=kwargs['max_value'])
        return v2

# Initialize the model
m = Model(min_value=-1, max_value=5)

# Inputs to the model
x2 = torch.randn(1, 8)
