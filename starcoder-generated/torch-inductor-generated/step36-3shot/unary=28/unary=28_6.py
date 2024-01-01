
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value=-0.5)
        v3 = torch.clamp_max(v2, max_value=0.4)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 4)
