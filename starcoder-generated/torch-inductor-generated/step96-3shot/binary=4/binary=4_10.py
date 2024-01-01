
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x2, x3):
        v1 = self.linear(x2)
        v2 = v1 + x3
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
x3 = torch.randn(5)
