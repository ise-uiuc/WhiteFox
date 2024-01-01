
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x):
        o1 = self.linear(x)
        o2 = o1 + 3
        o3 = torch.clamp_max(o2, 6)
        o4 = o3 / 6
        return o4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
