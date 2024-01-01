
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(8, 1)
 
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.l.weight, self.l.bias)
        v2 = torch.clamp(v1, min=0)
        v3 = torch.clamp(v2, max=6)
        div = v3 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
