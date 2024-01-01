
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1, __other__):
        x2 = self.linear(x1)
        x3 = x2 + __other__
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 3)
x2 = torch.randn(3, 3)
