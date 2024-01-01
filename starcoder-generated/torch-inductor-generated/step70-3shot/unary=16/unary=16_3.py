
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 32)
        self.linear2 = torch.nn.Linear(32, 8)
 
    def forward(self, x1):
        v2 = self.linear1(x1)
        v3 = torch.relu(v2)
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
