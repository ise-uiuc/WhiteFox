
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear2(x1)
        v2 = self.linear1(x1)
        v4 = torch.sigmoid(v1)
        v3 = v2 * v4
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
