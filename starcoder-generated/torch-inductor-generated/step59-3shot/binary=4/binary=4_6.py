
class Model(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.linear = torch.nn.Linear(shape[1], shape[0])
 
    def forward(self, x1, other):
        x2 = self.linear(x1)
        return x2 + other

# Initializing the model
m = Model([64, 3])

# Inputs to the model
x1 = torch.randn(64, 3)
other = torch.randn(64, 3)
