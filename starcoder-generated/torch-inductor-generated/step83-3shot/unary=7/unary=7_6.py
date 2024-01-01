

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(11, 12)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        c1 = torch.clamp(min=0, max=6, input=v1 + 3)
        v2 = v1 * c1 / 6
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 11)
