
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 2.718281828459045
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 128)
