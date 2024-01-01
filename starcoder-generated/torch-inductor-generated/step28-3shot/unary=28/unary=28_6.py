
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(1, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, min_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
min_value = 0.5
max_value = 1
x1 = torch.randn(1, 1)
