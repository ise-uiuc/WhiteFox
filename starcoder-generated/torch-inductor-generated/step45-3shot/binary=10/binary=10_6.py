
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x0, x1):
        v1 = self.linear(x0)
        v2 = v1 + x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 4)
x1 = torch.randn(1, 8)
