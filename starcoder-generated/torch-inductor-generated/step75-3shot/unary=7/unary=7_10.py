
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
 
    def forward(self, x2):
        v1 = self.l(x2)
        v2 = v1 * torch.clamp(torch.nn.functional.silu(v1) + 3, min=0., max=6.)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(2, 10)
