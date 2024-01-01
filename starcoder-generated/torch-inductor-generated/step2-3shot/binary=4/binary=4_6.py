
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 32)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        v1 = x2 + x1
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 10)
