
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        m1 = torch.clamp(min=0, max=6, v1 + 3)
        v9 = m1 / 6
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
