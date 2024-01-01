
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, a):
        v1 = self.linear(a)
        v2 = torch.clamp(v1 - 3, min=0, max=6)
        v3 = v2 * 6
        return v1

# Initializing the model
m = Model()

# Inputs to the model
a = torch.randn(128, 64)
