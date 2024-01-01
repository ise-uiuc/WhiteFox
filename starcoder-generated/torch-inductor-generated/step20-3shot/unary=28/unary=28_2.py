
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 10)
 
    def forward(self, x1):
        __input__ = torch.flatten(x1, start_dim = 1)
        v1 = self.fc(__input__)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 1)
        return v3
       
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
