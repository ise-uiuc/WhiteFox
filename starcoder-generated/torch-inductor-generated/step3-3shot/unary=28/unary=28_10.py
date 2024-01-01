
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x1, min_value=-1.0, max_value=-1.0):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        ret = torch.clamp_max(v2, max_value)
        return ret

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 2, 2)

# Parameters to the model
min_value = 2.0
max_value = 1.0

