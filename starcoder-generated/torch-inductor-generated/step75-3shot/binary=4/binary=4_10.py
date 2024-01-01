
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1, x2):
        v1 = self.linear(x2)
        v2 = v1 + 0.1
        v3 = v2 + x1
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 32, 64)
