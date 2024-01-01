
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(28, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.linear3 = torch.nn.Linear(256, 10)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = torch.add(v1, v2)
        v4 = self.linear3(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 28)
