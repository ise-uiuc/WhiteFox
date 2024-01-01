
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2


# Initializing the model
m = Model(torch.randn(1, 4))

# Inputs to the model
x = torch.randn(1, 4)
