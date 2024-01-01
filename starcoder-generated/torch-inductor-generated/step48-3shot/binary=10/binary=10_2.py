
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 16)
        self.linear2 = torch.nn.Linear(16, 9)

    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = v2 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 9)
