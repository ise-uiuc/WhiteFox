
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 8)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = F.relu(v1 + v2)
        v4 = v2 + v1
        return v3 + v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
