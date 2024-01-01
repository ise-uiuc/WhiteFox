
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 4)
 
    def forward(self, x, min_value, max_value):
        v1 = self.linear(x) + 1
        v2 = v1.clamp(min=min_value)
        v3 = v2.clamp_max(max_value)
        # v3
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 64)
min_value = -1.0
max_value = 2.0
