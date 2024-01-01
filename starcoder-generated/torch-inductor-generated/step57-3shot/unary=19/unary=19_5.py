
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)

    def forward(self, x1):
        x = self.linear(x1)
        y = torch.sigmoid(x)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
