
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
 
    def forward(self, l):
        a1 = self.linear(l)
        a2 = a1 * torch.clamp(a1 + 3, min=0, max=6)
        a3 = a2 / 6
        return a3

# Initializing the model
m = Model()

# Inputs to the model
l = torch.randn(1, 6)
