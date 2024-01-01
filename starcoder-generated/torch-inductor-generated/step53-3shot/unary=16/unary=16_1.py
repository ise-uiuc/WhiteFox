
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(384, 256)
        self.linear2 = nn.Linear(256, 384)
        self.linear3 = nn.Linear(384, 256)
        self.linear4 = nn.Linear(256, 42)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = F.relu(v1)
        v3 = self.linear2(v2)
        v4 = F.relu(v3)
        v5 = self.linear3(v4)
        v6 = F.relu(v5)
        v7 = self.linear4(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 384)
