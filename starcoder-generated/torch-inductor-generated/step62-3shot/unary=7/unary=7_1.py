
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        a1 = self.linear(x1)
        a2 = a1 * torch.clamp(a1 + 3, min=0, max=6)
        a3 = a2 / 6
        return a3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
