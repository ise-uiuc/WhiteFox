
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        __min_value__ = 0
        v2 = torch.clamp_min(v1, __min_value__)
        __max_value__ = 1
        v3 = torch.clamp_max(v2, __max_value__)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
