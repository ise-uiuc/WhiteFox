
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 8)
        self.linear5 = torch.nn.Linear(8, 4)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = self.linear3(v2)
        v4 = self.linear4(v3)
        v5 = self.linear5(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(64, 3, 64, 64)
