
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(42, 24)
 
    def forward(self, x1):
        v2 = self.linear(x1)
        v2.add_(other)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 42)
