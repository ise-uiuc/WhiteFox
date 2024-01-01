
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)
 
    def forward(self, x0):
        v0 = self.linear(x0)
        v1 = v0 > 0
        v2 = v0 * 0.01
        v3 = torch.where(v1, v0, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(3, 100)
