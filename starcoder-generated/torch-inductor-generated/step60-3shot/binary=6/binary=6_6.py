
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(100, 10)
 
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = v1 - 2.0
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100)
