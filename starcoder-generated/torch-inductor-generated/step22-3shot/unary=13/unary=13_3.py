
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 1000)
 
    def forward(self, v1):
        v2 = self.linear(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 1000)
