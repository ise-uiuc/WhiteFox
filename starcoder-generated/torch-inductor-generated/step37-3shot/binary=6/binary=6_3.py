
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)
 
    def forward(self, x1):
        return self.linear(x1) - 10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 100)
