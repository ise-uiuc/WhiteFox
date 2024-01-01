
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = self.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x_rand = torch.randn(4, 5)
