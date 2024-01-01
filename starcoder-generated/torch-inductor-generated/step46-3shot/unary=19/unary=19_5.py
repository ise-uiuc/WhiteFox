
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(192, 64)
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 8)
        self.linear4 = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = self.linear0(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.linear1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.linear2(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.linear3(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.linear4(v8)
        v10 = torch.sigmoid(v9)
        return v10

# Initializing the model
m = Model()

# Initializing the inputs to the model
x1 = torch.randn(1, 192)
