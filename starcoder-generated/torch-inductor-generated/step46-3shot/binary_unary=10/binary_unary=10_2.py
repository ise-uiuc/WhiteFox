
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.rand_like(v1)
        v3 = v1 + v2
        v4 = F.relu(v3)
        v5 = self.linear2(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
