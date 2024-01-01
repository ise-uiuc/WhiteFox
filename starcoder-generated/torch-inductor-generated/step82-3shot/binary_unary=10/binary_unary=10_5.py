
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 1)
        self.linear2 = nn.Linear(1, 1)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + self.linear2.weight
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
