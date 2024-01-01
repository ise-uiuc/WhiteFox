
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 9707)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = other
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(256)
