
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32)
        self.linear2 = torch.nn.Linear(32, 48)
        self.linear3 = torch.nn.Linear(48, 32)
        self.linear4 = torch.nn.Linear(32, 16)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.tanh(v1)
        v3 = self.linear2(v2)
        v4 = torch.tanh(v3)
        v5 = self.linear3(v4)
        v6 = torch.tanh(v5)
        v7 = self.linear4(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(70, 16)
