
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 128)
 
    def forward(self, x1):
        o1 = self.linear(x1)
        o2 = o1 + 3
        o3 = torch.clamp_min(o2, 0)
        o4 = torch.clamp_max(o3, 6)
        o5 = o4 / 6
        return o5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
