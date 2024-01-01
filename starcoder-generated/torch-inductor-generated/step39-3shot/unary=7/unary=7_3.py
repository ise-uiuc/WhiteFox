
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2000, 4000)
 
    def forward(self, x1):
        o1 = self.linear(x1)
        o2 = o1 * torch.clamp(o1, min=0, max=6) + 3
        o3 = o2 / 6
        return o3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2000)
