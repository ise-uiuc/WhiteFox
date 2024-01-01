
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 2
        v3 = v1 - 3
        v4 = v3 - 4
        v5 = v3 - 8
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 16)
