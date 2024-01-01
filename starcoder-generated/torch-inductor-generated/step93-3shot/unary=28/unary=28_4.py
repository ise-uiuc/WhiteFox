
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 2.5)
        v3 = torch.clamp_max(v2, 6.5)
        return v3

# Initializing the model
x1 = __input_tensor__(8, 8)
m = Model()

# Inputs to the model
