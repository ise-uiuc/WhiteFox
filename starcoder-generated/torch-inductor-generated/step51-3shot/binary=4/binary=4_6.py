
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(21, 26)
 
    def forward(self, x2, x3):
        v2 = self.linear(x2)
        v3 = v2 + x3
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 21, 1)
x3 = torch.randn(1, 26, 1)
