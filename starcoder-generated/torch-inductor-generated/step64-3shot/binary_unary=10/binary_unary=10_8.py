
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(19, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = self.relu(v2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 19)
