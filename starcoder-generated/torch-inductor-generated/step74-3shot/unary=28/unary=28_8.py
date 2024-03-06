
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 0.0)
        v3 = torch.clamp_max(v2, 10.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
min_value = 0.0
max_value = 10.0