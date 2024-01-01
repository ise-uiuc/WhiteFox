
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 1, bias=False)
        self.linear2 = torch.nn.Linear(1, 2, bias=False)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - 1.0
        v3 = F.relu(v2)
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
