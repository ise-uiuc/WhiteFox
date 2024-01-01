
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = x1.shape
        v3 = v2 - 2
        return v1 + v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
