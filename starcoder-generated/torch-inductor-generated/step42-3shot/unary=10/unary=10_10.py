
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1):
        y1 = self.linear(x1)
        return F.relu6(y1 + 3) * (6. / 6.)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
