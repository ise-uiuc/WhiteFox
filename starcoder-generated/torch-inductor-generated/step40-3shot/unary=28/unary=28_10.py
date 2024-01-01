
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = F.clamp_min(v1, min_value=-2)
        v3 = F.clamp_max(v2, max_value=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 4)
