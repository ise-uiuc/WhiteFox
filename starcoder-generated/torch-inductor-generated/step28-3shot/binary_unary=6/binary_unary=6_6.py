
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 100)
        self.linear2 = torch.nn.Linear(100, 1)

    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 10)
x2 = torch.randn(128, 10)
