
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(-1, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - v2
        return v1, v2, v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
v2 = torch.randn(1, 5)
v3 = 10
