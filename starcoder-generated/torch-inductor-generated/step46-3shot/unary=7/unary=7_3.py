
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(786, 512)
        self.clamp = torch.nn.Identity()
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = self.clamp(x2 + 3)
        x4 = x3 / 6
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 786)
