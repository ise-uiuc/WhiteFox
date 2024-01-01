
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x1, min_value, max_value):
        v1 = self.linear(x1)
        t2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(t2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
min_value = 0.05
max_value = 0.9
