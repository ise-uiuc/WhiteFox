
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1, x2):
        v1 = self.linear(x2)
        v2 = v1 - 10
        v3 = F.relu(v1)
        v4 = F.relu(v2)
        v5 = F.relu(v3)
        v6 = F.relu(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
