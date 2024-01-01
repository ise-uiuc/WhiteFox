
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
 
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=self.min_value)
        v3 = torch.clamp(v2, max=self.max_value)
        return v3

# Initializing the model
m = Model(min_value=0.0, max_value=0.8)

# Inputs to the model
x1 = torch.randn(1, 8)
