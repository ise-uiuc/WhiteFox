
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        o1 = self.linear(x1)
        o2 = o1 + 3
        o3 = torch.clamp(o2, 0, 6)
        o4 = o3 / 6
        return o4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(6, 3)
