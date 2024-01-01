
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)
 
    def forward(self, x1, x2):
        v2 = self.linear(x1)
        v3 = v2 + x2
        return v3, v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 5)
x2 = torch.randn(2, 2)
