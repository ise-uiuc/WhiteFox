
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(25, 50)

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2)
        v2 = self.linear1(v1)
        v3 = torch.cat([v1, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 25)
x2 = torch.randn(25, 50)
