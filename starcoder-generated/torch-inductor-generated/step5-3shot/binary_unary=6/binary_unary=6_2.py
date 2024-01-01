
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 32)
        self.linear2 = torch.nn.Linear(32, 64)
        self.linear3 = torch.nn.Linear(64, 128)
 
    def forward(self, x2):
        v4 = x2.view((1, 8))
        v2 = self.linear1(v4)
        v5 = v2 - 1000.0
        v6 = torch.relu(v5)
        v3 = self.linear2(v6)
        v8 = v3 - 2000.0
        v7 = torch.relu(v8)
        v1 = self.linear3(v7)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
