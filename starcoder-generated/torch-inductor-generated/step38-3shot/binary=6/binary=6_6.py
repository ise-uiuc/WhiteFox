
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 0.5
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 5)
