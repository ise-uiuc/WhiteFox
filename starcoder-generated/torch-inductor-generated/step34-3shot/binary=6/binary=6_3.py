
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(self, x):
        x1 = self.linear(x)
        return x1 - 3.14

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
