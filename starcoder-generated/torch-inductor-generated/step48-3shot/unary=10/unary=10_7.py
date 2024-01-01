
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 3)
        self.linear3 = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + 3
        v3 = torch.max(torch.min(v2, torch.tensor(6.0)), torch.tensor(0.0))
        v4 = v3 / 6
        v5 = self.linear2(v4)
        v6 = self.linear3(v5)
        return v6
    
# Initializing the model
__m__ = Model()

# Inputs to the model
__x1__ = torch.randn(1, 3)
