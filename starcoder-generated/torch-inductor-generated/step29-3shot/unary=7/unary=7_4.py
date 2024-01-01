
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(min=0, max=6, v1)
        v3 = v2 + 3
        return v3 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(224, 224)
