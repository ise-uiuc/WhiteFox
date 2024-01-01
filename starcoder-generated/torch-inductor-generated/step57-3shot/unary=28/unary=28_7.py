
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=-0.4)
        v3 = torch.clamp(v2, max=0.4)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8, 4, 4)
