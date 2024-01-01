
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 256 * 256, 1)
 
    def forward(self, x1):
        v3 = x1.flatten(1)
        v3 = self.linear(v3)
        v2 = v3 - 1
        v1 = torch.relu(v2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 256, x2)
