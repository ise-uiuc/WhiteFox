
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        x3 = torch.cat([x1, x2], dim=1)
        v1 = self.linear(x3)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 1, 64, 64)
x2 = torch.randn(3, 1, 64, 64)
