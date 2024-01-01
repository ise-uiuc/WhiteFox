
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x1):
        a1 = self.linear(x1)
        a2 = a1 + 3
        a3 = torch.clamp_min(a2, 0)
        a4 = torch.clamp_max(a3, 6)
        a5 = a4 / 6
        return a5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
