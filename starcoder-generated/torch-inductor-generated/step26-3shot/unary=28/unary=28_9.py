
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
def forward(self, x0):
    v0 = self.linear(x1)
    v1 = torch.clamp_min(v0, min_value=0)
    return torch.clamp_max(v1, max_value=24)

# Initializing the model
