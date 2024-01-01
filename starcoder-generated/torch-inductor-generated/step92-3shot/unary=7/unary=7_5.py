
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 3)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = v7 + 3
        v9 = v8.clamp(min=0, max=6)
        v10 = v9 / 6
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 16)
