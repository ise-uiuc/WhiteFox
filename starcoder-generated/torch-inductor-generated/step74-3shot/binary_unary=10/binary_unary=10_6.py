
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 16)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 2
        v3 = v1 + 5
        v4 = v3 * v2
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 10)
