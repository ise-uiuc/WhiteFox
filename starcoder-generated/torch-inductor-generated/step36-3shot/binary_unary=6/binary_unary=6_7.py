
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 256)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.56
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 100)
