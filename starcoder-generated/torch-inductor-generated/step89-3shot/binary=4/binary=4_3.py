
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 768)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        return v1 + other

# Initializing the model and the addend tensor
m = Model()
addend = torch.full((1, 768), -1)

# Inputs to the model
x1 = torch.randn(1, 32)
