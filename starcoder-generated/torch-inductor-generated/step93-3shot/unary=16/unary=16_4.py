
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = torch.nn.Linear(8, 8)
        self.d2 = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.d1(x1)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 6, 6)
