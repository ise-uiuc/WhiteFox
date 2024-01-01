
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x0):
        v0 = self.linear(x0)
        v2 = torch.sigmoid(v0)
        v3 = v0 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 8)
