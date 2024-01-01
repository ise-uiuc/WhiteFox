
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 1)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1.reshape(1, -1)
        v3 = self.linear2(v2)
        v4 = self.linear3(v3)
        v5 = torch.sigmoid(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
