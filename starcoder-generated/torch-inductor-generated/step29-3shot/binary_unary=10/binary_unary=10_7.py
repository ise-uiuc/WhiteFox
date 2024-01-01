
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        v4 = self.linear(v3)
        v5 = v4 + x2
        v6 = v4 * v5
        v7 = v3 - v6
        v8 = self.linear(v7)
        v9 = v8 * 0.3
        v10 = torch.abs(v9)
        v11 = F.sigmoid(v10)
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
x3 = torch.randn(1, 3)
