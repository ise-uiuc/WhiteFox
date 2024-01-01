
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 512)
 
    def forward(self, x2, x3, x4):
        v1 = self.linear(x2)
        v2 = torch.nn.functional.relu(v1)
        v3 = v2 + x3
        v4 = v3 * x4
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(20, 3)
