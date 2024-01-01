
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.linear2(v3)
        v5 = v4 * v2
        v6 = v4 + v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
