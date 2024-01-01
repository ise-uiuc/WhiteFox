
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        r1 = self.linear(x1)
        r2 = torch.clamp(r1 + 3, min=0, max=6)
        r3 = r2 / 6
        return r3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
