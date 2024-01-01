
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        r1 = self.linear(x)
        r2 = torch.clamp(r1, min=0, max=6.0)
        r3 = self.linear(r1) / 6.0
        return r1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
