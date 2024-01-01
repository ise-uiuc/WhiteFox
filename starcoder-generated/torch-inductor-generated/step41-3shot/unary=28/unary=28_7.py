
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=-3, max=12)
        v3 = torch.clamp(v2, min=0, max=6)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
