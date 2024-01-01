
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=0)
        v3 = torch.clamp_max(v2, max=10)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 3)
min_value = -1.0
max_value = 5.0
