
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model which adds x1 to the model output
m1 = Model(x1)

# Inputs to the model
x2 = torch.randn(1, 8)
