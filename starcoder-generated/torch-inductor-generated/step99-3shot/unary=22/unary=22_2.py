
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        ret = self.linear(x)
        return torch.tanh(ret)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 10)
