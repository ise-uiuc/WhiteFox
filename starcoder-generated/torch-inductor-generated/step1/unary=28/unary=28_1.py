
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x, min_value, max_value):
        v1 = self.linear(x)
        v2 = v1.clamp_min(min_value)
        v3 = v2.clamp_max(max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
min_value = 2
max_value = 6
