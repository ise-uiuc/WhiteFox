
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        v4 = self.linear(v3)
        v5 = v4 + x3
        v6 = F.relu(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
x2 = torch.randn(1, 6)
x3 = torch.randn(1, 6)
