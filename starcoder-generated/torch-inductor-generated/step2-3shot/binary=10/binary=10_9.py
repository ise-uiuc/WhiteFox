
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 2)
        self.linear2 = torch.nn.Linear(2, 5)
    
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = torch.relu(v2) - 1.5
        v4 = torch.relu(v3) - 2.0
        v5 = v4 * 6.5
        v6 = v1 / torch.sigmoid(v5)
        v7 = v6 % x2
        v8 = v1 * v6
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 5)
