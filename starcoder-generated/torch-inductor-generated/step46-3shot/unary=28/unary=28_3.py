
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=-100)
        v3 = torch.clamp_max(v2, max_value=100)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
min_value = -0.3
max_value = 0.8
x1 = torch.randn(1, 10)
