
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(8, 2)

    def forward(self, x1):
        return self.layer(torch.tanh(x1))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
