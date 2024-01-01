
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        v2 = torch.nn.functional.relu(self.linear(x))
        v3 = torch.clamp_min(v2, min_value=min_value)
        v1 = torch.clamp_max(v3, max_value=max_value)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)

