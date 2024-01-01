
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        max_value = 20
        v2 = torch.clamp_min(v1, -max_value)
        min_value = -40
        v3 = torch.clamp_max(v2, min_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
