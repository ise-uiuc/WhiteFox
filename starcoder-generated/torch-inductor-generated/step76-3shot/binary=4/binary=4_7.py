
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 10, bias=True)

    def forward(self, x1):
        y1 = self.linear1(x1)
        v1 = torch.tanh(y1)
        z1 = torch.sigmoid(v1)
        return z1

# Initializing the model
m = Model()

# Inputs to the model.
x1 = torch.randn(1, 3)
