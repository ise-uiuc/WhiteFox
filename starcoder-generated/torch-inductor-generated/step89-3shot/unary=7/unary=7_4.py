
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16, bias=False)
 
    def forward(self, x1):
        v1 = torch.clamp(self.linear(x1), min=0., max=6.)
        v2 = torch.mul(v1, 0.166667)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
