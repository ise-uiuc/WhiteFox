
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x, max_value):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value=0)
        v3 = torch.clamp_max(v2, max_value=max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
max_value = 6.5
